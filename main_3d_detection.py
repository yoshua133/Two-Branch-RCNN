import csv
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse

from cuda_functions.nms_2D.pth_nms import nms_gpu as nms_2D
from cuda_functions.nms_3D.pth_nms import nms_gpu as nms_3D
from cuda_functions.roi_align_2D.roi_align.crop_and_resize import CropAndResizeFunction as ra2D
from cuda_functions.roi_align_3D.roi_align.crop_and_resize import CropAndResizeFunction as ra3D



import sys
import mutil as mutils
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from default_configs import DefaultConfigs


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str,default="/home/dxiang/3d_detection/output/")



parm = parser.parse_args()

class configure(DefaultConfigs):
    def __init__(self, server_env=None):
        #########################
        #    Preprocessing      #
        #########################
        self.root_dir = '/path/to/raw/data'
        self.raw_data_dir = '{}/data_nrrd'.format(self.root_dir)
        self.pp_dir = '{}/pp_norm'.format(self.root_dir)
        self.target_spacing = (0.7, 0.7, 1.25)
        #########################
        #         I/O           #
        #########################
        # one out of [2, 3]. dimension the model operates in.
        self.sixth_pooling=False
        self.dim1 = 3
        self.dim=2
        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_unet', 'ufrcnn', 'detection_unet'].
        self.model = 'mrcnn'
        DefaultConfigs.__init__(self, self.model, server_env, self.dim)
        # int [0 < dataset_size]. select n patients from dataset for prototyping. If None, all data is used.
        self.select_prototype_subset = None
        # path to preprocessed data.
        self.pp_name = 'pp_norm'
        self.input_df_name = 'info_df.pickle'
        self.pp_data_path = '/path/to/preprocessed/data/{}'.format(self.pp_name)
        self.pp_test_data_path = self.pp_data_path #change if test_data in separate folder.
        # settings for deployment in cloud.
        if server_env:
            # path to preprocessed data.taide6953
            self.pp_name = 'pp_fg_slices'
            self.crop_name = 'pp_fg_slices_packed'
            self.pp_data_path = '/path/to/preprocessed/data/{}/{}'.format(self.pp_name, self.crop_name)
            self.pp_test_data_path = self.pp_data_path
            self.select_prototype_subset = None
        #########################
        #      Data Loader      #
        #########################
        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)
        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size_2D = [300, 300]
        self.patch_size_2D = [512, 512]
        self.pre_crop_size_3D = [156, 156, 96]
        self.patch_size_3D = [128, 128, 64]
        self.patch_size = self.patch_size_2D if self.dim == 2 else self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_2D if self.dim == 2 else self.pre_crop_size_3D
        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_sample_slack = 0.2
        # set 2D network to operate in 3D images.
        self.merge_2D_to_3D_preds = True
        # feed +/- n neighbouring slices into channel dimension. set to None for no context.
        self.n_3D_context = None
        if self.n_3D_context is not None and self.dim == 2:
            self.n_channels *= (self.n_3D_context * 2 + 1)
        #########################
        #      Architecture      #
        #########################
        self.start_filts = 48 if self.dim == 2 else 18
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101' , 'resnet50'
        self.norm = None # one of None, 'instance_norm', 'batch_norm'
        self.weight_decay = 0
        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (=default = 'kaiming_uniform')
        self.weight_init = None
        #########################
        #  Schedule / Selection #
        #########################
        self.num_epochs = 100
        self.num_train_batches = 200 if self.dim == 2 else 200
        self.batch_size = 3 if self.dim == 2 else 8
        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is morge accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling' , 'val_patient'
        if self.val_mode == 'val_patient':
            self.max_val_patients = 50  # if 'None' iterates over entire val_set once.
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 50
        #########################
        #   Testing / Plotting  #
        #########################
        # set the top-n-epochs to be saved for temporal averaging in testing.
        self.save_n_models = 5
        self.test_n_epochs = 5
        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0
        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1: 'benign', 2: 'malignant'}  # 0 is background.
        self.patient_class_of_interest = 2  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.1]  # list of ious to be evaluated for ap-scoring.
        self.model_selection_criteria = ['malignant_ap', 'benign_ap'] # criteria to average over for saving epochs.
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.
        # threshold for clustering predictions together (wcs = weighted cluster scoring).
        # needs to be >= the expected overlap of predictions coming from one model (typically NMS threshold).
        # if too high, preds of the same object are separate clusters.
        self.wcs_iou = 1e-5
        self.plot_prediction_histograms = True
        self.plot_stat_curves = False
        #########################
        #   Data Augmentation   #
        #########################
        self.da_kwargs={
        'do_elastic_deform': True,
        'alpha':(0., 1500.),
        'sigma':(30., 50.),
        'do_rotation':True,
        'angle_x': (0., 2 * np.pi),
        'angle_y': (0., 0),
        'angle_z': (0., 0),
        'do_scale': True,
        'scale':(0.8, 1.1),
        'random_crop':False,
        'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
        'border_mode_data': 'constant',
        'border_cval_data': 0,
        'order_data': 1
        }
        if self.dim == 3:
            self.da_kwargs['do_elastic_deform'] = False
            self.da_kwargs['angle_x'] = (0, 0.0)
            self.da_kwargs['angle_y'] = (0, 0.0) #must be 0!!
            self.da_kwargs['angle_z'] = (0., 2 * np.pi)
        #########################
        #   Add model specifics #
        #########################
        {'detection_unet': self.add_det_unet_configs,
         'mrcnn': self.add_mrcnn_configs,
         'ufrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
        }[self.model]()
    def add_det_unet_configs(self):
        self.learning_rate = [1e-4] * self.num_epochs
        # aggregation from pixel perdiction to object scores (connected component). One of ['max', 'median']
        self.aggregation_operation = 'max'
        # max number of roi candidates to identify per batch element and class.
        self.n_roi_candidates = 10 if self.dim == 2 else 30
        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'dice_wce'
        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1
        self.wce_weights = [1, 1, 1]
        self.detection_min_confidence = self.min_det_thresh
        # if 'True', loss distinguishes all classes, else only foreground vs. background (class agnostic).
        self.class_specific_seg_flag = True
        self.num_seg_classes = 9 if self.class_specific_seg_flag else 2
        self.head_classes = self.num_seg_classes
    def add_mrcnn_configs(self):
        # learning rate is a list with one entry per epoch.
        self.learning_rate = [1e-4] * self.num_epochs
        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_val = True
        self.return_masks_in_test = False
        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 5 if self.dim == 2 else 30
        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = 10
        # seg_classes hier refers to the first stage classifier (RPN)
        self.num_seg_classes = 9  # foreground vs. background
        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}
        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]}
        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]
        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 128
        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5, 1, 2]
        self.rpn_anchor_stride = 1
        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7
        # loss sampling settings.
        self.rpn_train_anchors_per_image = 6  #per batch element
        self.train_rois_per_image = 6 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7
        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10
        self.pool_size = (7, 7) #if self.dim == 2 else (7, 7, 3)
        self.mask_pool_size = (14, 14) #if self.dim == 2 else (14, 14, 5)
        self.mask_shape = (28, 28) #if self.dim == 2 else (28, 28, 10)
        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size_3D[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1],
                               self.patch_size_3D[2], self.patch_size_3D[2]])
        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]
        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000
        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 500 if self.dim == 2 else 75
        self.post_nms_rois_inference = 500
        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 10 if self.dim == 2 else 30  # per batch element and class.
        self.detection_nms_threshold = 1e-5  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1
        if self.dim == 2:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])
        else:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride)),
                  int(np.ceil(self.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z']
                                             )])
        if self.model == 'ufrcnn':
            self.operate_stride1 = True
            self.class_specific_seg_flag = True
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            self.frcnn_mode = True
        if self.model == 'retina_net' or self.model == 'retina_unet' or self.model == 'prob_detector':
            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3
            self.n_rpn_features = 256 if self.dim == 2 else 64
            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 50000
            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5
            # if 'True', seg loss distinguishes all classes, else only foreground vs. background (class agnostic).
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            if self.model == 'retina_unet':
                self.operate_stride1 = True

cf=configure()
print(cf.backbone_shapes)

def load_data(path):
      data=cv2.imread(path,-1)
      return data


class dataset_3d(torch.utils.data.Dataset):
  def __init__(self,
  image_root="/data/deeplesion/images/Images_png/",
  csv_paths="/home/dxiang/DL_info_liver.csv"):
    self.image_root=image_root
    self.csv_paths=csv_paths
    self.File_name=[]
    self.Key_slice_index=[]
    self.Bounding_boxes=[]	
    self.Coarse_lesion_type=[]	
    self.Slice_range=[]
    self.Spacing_mm_px_=[]	
    self.Image_size=[]	
    self.DICOM_windows=[]
    self.slice_length=[]	
    self.total_size=[]
    self.slice_5=[]
    self.thick=[]
    #setting parameter
    self.input_depth=8
    self.mm_per_slice=5
    with open(csv_paths, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = list(csv_reader)
        for row in rows:
            if row["Coarse_lesion_type"]=='4' :#which means using the image that identified as liver images
                #print(row["File_name"])
                self.Key_slice_index.append(row["Key_slice_index"])
                self.File_name.append(row["File_name"])
                bb=row["Bounding_boxes"]
                bbox=np.array(bb.split(','))
                self.Bounding_boxes.append(bbox.astype(np.float))
                self.Coarse_lesion_type.append(row["Coarse_lesion_type"])
                self.Slice_range.append(row["Slice_range"])
                self.Spacing_mm_px_.append(row["Spacing_mm_px_"])
                self.Image_size.append(row["Image_size"])
                self.DICOM_windows.append(row["DICOM_windows"])
                min,max=row["Slice_range"].split(',')
                space=row["Spacing_mm_px_"].split(',')
                thick=float(space[2])
                self.slice_length.append(float(max)-float(min))   
                self.total_size.append(thick*(float(max)-float(min)))
                self.thick.append(thick)
  def get_slice_name(self,File_name,slice_id):
      file_name=File_name.split('_')
      folder_name=file_name[0]+'_'+file_name[1]+'_'+file_name[2]
      slice_name=str(slice_id).zfill(3)+'.png'
      slice_path=os.path.join(self.image_root,folder_name,slice_name)
      return slice_path
  def __getitem__(self,index):
      output=np.zeros([1,self.input_depth,512,512])
      center_slice_id=int(self.Key_slice_index[index])
      step=int(self.thick[index]/self.mm_per_slice)
      center_file_name=self.get_slice_name(self.File_name[index],center_slice_id)
      output[:,int(self.input_depth/2),:,:]=load_data(center_file_name)
      upper_slice_path=self.get_slice_name(self.File_name[index],center_slice_id+step)
      lower_slice_path=self.get_slice_name(self.File_name[index],center_slice_id-step)
      upper_place=1
      #print(upper_slice_path)
      #print(lower_slice_path)
      while(os.path.exists(upper_slice_path)and(upper_place<self.input_depth/2)):
          output[:,int(self.input_depth/2)+upper_place,:,:]=load_data(upper_slice_path)
          #print(upper_place)
          #print(upper_slice_path)
          upper_place+=1
          upper_slice_path=self.get_slice_name(self.File_name[index],center_slice_id+upper_place*step)
      lower_place=1
      while(os.path.exists(lower_slice_path)and(lower_place<self.input_depth/2+1)):
          #print(lower_slice_path)
          output[:,int(self.input_depth/2)-lower_place,:,:]=load_data(lower_slice_path)
          lower_place+=1
          lower_slice_path=self.get_slice_name(self.File_name[index],center_slice_id-lower_place*step)
      return output,self.Bounding_boxes[index],upper_place,lower_place,self.Coarse_lesion_type[index]
  def __len__(self):
    return len(self.File_name)
    


def proposal_layer(rpn_pred_probs, rpn_pred_deltas, proposal_count, anchors, cf):
    """
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinment detals to anchors.
    :param rpn_pred_probs: (b, n_anchors, 2)
    :param rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
    :return: batch_normalized_boxes: Proposals in normalized coordinates
    (b, proposal_count, (y1, x1, y2, x2, (z1), (z2)))
    :return: batch_out_proposals: Box coords + RPN foreground scores
    for monitoring/plotting (b, proposal_count, (y1, x1, y2, x2, (z1), (z2), score))
    """
    batch_scores = rpn_pred_probs[:, :, 1]
    batch_deltas = rpn_pred_deltas
    batch_anchors = anchors
    batch_normalized_boxes = []
    batch_out_proposals = []

    # loop over batch dimension.
    for ix in range(batch_scores.shape[0]):

        scores = batch_scores[ix]
        deltas = batch_deltas[ix]
        anchors = batch_anchors.clone()
        # norm deltas
        std_dev = torch.from_numpy(cf.rpn_bbox_std_dev[None]).float().cuda()
        deltas = deltas * std_dev

        # improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = min(cf.pre_nms_limit, anchors.size()[0])
        scores, order = scores.sort(descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = deltas[order, :]
        anchors = anchors[order, :]

        # apply deltas to anchors to get refined anchors and filter with non-maximum surpression.
        if batch_deltas.shape[-1] == 4:
            boxes = mutils.apply_box_deltas_2D(anchors, deltas)
            boxes = mutils.clip_boxes_2D(boxes, cf.window)
            keep = nms_2D(torch.cat((boxes, scores.unsqueeze(1)), 1), cf.rpn_nms_threshold)
            norm = torch.from_numpy(cf.scale).float().cuda()

        else:
            boxes = mutils.apply_box_deltas_3D(anchors, deltas)
            boxes = mutils.clip_boxes_3D(boxes, cf.window)
            keep = nms_3D(torch.cat((boxes, scores.unsqueeze(1)), 1), cf.rpn_nms_threshold)
            norm = torch.from_numpy(cf.scale).float().cuda()

        keep = keep[:proposal_count]
        boxes = boxes[keep, :]
        rpn_scores = scores[keep][:, None]

        # padding missing boxes with 0.
        if boxes.shape[0] < proposal_count:
            n_pad_boxes = proposal_count - boxes.shape[0]
            zeros = torch.zeros([n_pad_boxes, boxes.shape[1]]).cuda()
            boxes = torch.cat([boxes, zeros], dim=0)
            zeros = torch.zeros([n_pad_boxes, rpn_scores.shape[1]]).cuda()
            rpn_scores = torch.cat([rpn_scores, zeros], dim=0)

        # concat box and score info for monitoring/plotting.
        batch_out_proposals.append(torch.cat((boxes, rpn_scores), 1).cpu().data.numpy())
        # normalize dimensions to range of 0 to 1.
        normalized_boxes = boxes / norm
        # add back batch dimension
        batch_normalized_boxes.append(normalized_boxes.unsqueeze(0))

    batch_normalized_boxes = torch.cat(batch_normalized_boxes)
    batch_out_proposals = np.array(batch_out_proposals)
    return batch_normalized_boxes, batch_out_proposals

class NDConv(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim
    def __call__(self, c_in, c_out, kernel_size, padding=0, stride=1, norm=None, relu='relu'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param kernel_size: kernel size.
        :param padding: padding size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)
        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)
        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)
        return conv


class ResBlock(nn.Module):
    def __init__(self, start_filts, planes, conv, stride=1, norm=None, downsample=None, relu='relu'):
        super(ResBlock, self).__init__()
        self.conv1 = conv(start_filts, planes, kernel_size=1, stride=stride)
        self.conv2 = conv(planes, planes, kernel_size=3, padding=1)
        self.conv3 = conv(planes, planes * 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True) if relu == 'relu' else nn.LeakyReLU(inplace=True)
        if downsample is not None:
            self.downsample = conv(downsample[0], downsample[0] * downsample[1], kernel_size=1, stride=downsample[2])
        else:
            self.downsample = None
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cf, conv, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param start_filts:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()
        self.start_filts = cf.start_filts
        start_filts = self.start_filts
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cf.res_architecture], 3]
        self.block = ResBlock
        self.block_expansion = 4
        self.operate_stride1 = operate_stride1
        self.sixth_pooling = cf.sixth_pooling
        self.dim = 3
        if operate_stride1:
            self.C0 = nn.Sequential(conv(cf.n_channels, start_filts, kernel_size=3, padding=1),
                                    conv(start_filts, start_filts, kernel_size=3, padding=1))
            self.C1 = conv(start_filts, start_filts, kernel_size=7, stride=(1, 2, 2) if self.dim == 3 else 2, padding=3)
        else:
            self.C1 = conv(cf.n_channels, start_filts, kernel_size=7, stride=(1, 2, 2) if self.dim == 3 else 2, padding=3)
        start_filts_exp = start_filts * self.block_expansion
        C2_layers = []
        C2_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                         if self.dim == 2 else nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1))
        C2_layers.append(self.block(start_filts, start_filts, conv=conv, stride=1, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts, self.block_expansion, 1)))
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(start_filts_exp, start_filts, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C2 = nn.Sequential(*C2_layers)
        C3_layers = []
        C3_layers.append(self.block(start_filts_exp, start_filts * 2, conv=conv, stride=2, norm=cf.norm, relu=cf.relu,
                                    downsample=(start_filts_exp, 2, 2)))
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(start_filts_exp * 2, start_filts * 2, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C3 = nn.Sequential(*C3_layers)
        C4_layers = []
        C4_layers.append(self.block(
            start_filts_exp * 2, start_filts * 4, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 2, 2, 2)))
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(start_filts_exp * 4, start_filts * 4, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C4 = nn.Sequential(*C4_layers)
        C5_layers = []
        C5_layers.append(self.block(
            start_filts_exp * 4, start_filts * 8, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 4, 2, 2)))
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(start_filts_exp * 8, start_filts * 8, conv=conv, norm=cf.norm, relu=cf.relu))
        self.C5 = nn.Sequential(*C5_layers)
        if self.sixth_pooling:
            C6_layers = []
            C6_layers.append(self.block(
                start_filts_exp * 8, start_filts * 16, conv=conv, stride=2, norm=cf.norm, relu=cf.relu, downsample=(start_filts_exp * 8, 2, 2)))
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(start_filts_exp * 16, start_filts * 16, conv=conv, norm=cf.norm, relu=cf.relu))
            self.C6 = nn.Sequential(*C6_layers)
        # # if self.dim == 2:
            # # self.P1_upsample = F.interpolate(scale_factor=2, mode='bilinear')
            # # self.P2_upsample = F.interpolate(scale_factor=2, mode='bilinear')
        # # else:
            # # self.P1_upsample = F.interpolate(scale_factor=(2, 2, 1), mode='trilinear')
            # # self.P2_upsample = F.interpolate(scale_factor=(2, 2, 1), mode='trilinear')
        self.out_channels = cf.end_filts
        Relu = nn.LeakyReLU(0.01)
        self.P5_conv1 = conv(start_filts*32 + cf.n_latent_dims, self.out_channels, kernel_size=1, stride=1) #
        self.P4_conv1 = conv(start_filts*16, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv1 = conv(start_filts*8, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv1 = conv(start_filts*4, self.out_channels, kernel_size=1, stride=1)
        self.P1_conv1 = conv(start_filts, self.out_channels, kernel_size=1, stride=1)
        if operate_stride1:
            self.P0_conv1 = conv(start_filts, self.out_channels, kernel_size=1, stride=1)
            self.P0_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P1_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P2_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P3_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P4_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.P5_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.sixth_pooling:
            self.P6_conv1 = conv(start_filts * 64, self.out_channels, kernel_size=1, stride=1)
            self.P6_conv2 = conv(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        """
        :param x: input image of shape (b, c, y, x, (z))
        :return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x
        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        # if self.sixth_pooling:
            # c6_out = self.C6(c5_out)
            # p6_pre_out = self.P6_conv1(c6_out)
            # p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        # else:
        p5_pre_out = self.P5_conv1(c5_out)
        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        #print("p4_out",p4_pre_out.shape)
        #print("c3_out",c3_out.shape)
        #print("p3_out",self.P3_conv1(c3_out).shape)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)
        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]
        # if self.sixth_pooling:
            # p6_out = self.P6_conv2(p6_pre_out)
            # out_list.append(p6_out)
        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.F.interpolate(p2_pre_out,scale_factor=(2, 2, 1), mode='trilinear')
            p0_pre_out = self.P0_conv1(c0_out) + self.F.interpolate(p1_pre_out,scale_factor=(2, 2, 1), mode='trilinear')
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list
                # plot feature map shapes for debugging.
        #for ii in [c2_out, c3_out, c4_out, c5_out]:
             #print ("encoder shapes:", ii.shape)
        #for ii in [p5_out, p4_out, p3_out, p2_out]:
             #print("decoder shapes:", ii.shape)
        return out_list

class RPN(nn.Module):
    """
    Region Proposal Network.
    """
    def __init__(self, cf, conv,conv2):
        super(RPN, self).__init__()
        self.dim =2
        self.conv_shared2= conv2(cf.end_filts*32, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_shared3 = conv2(cf.end_filts*16, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_shared4 = conv2(cf.end_filts*8, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_shared5 = conv2(cf.end_filts*4, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_shared6 = conv2(cf.end_filts*2, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_shared7 = conv2(cf.end_filts, cf.n_rpn_features, kernel_size=3, stride=cf.rpn_anchor_stride, padding=1)
        self.conv_class = conv2(cf.n_rpn_features, 2 * len(cf.rpn_anchor_ratios), kernel_size=1, stride=1)#relu
        self.conv_bbox = conv2(cf.n_rpn_features, 2 * self.dim * len(cf.rpn_anchor_ratios), kernel_size=1, stride=1)
    def forward(self, x):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :return: rpn_class_logits (b, 2, n_anchors)
        :return: rpn_probs_logits (b, 2, n_anchors)
        :return: rpn_bbox (b, 2 * dim, n_anchors)
        """
        filts=x.shape[1]*x.shape[2]
        x=x.view(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4])
        #print("x orig shape",x.shape)
        if filts == cf.end_filts*32:
            x=self.conv_shared2(x)
        elif filts == cf.end_filts*16:
            x=self.conv_shared3(x)
        elif filts == cf.end_filts*8:
            x=self.conv_shared4(x)
        elif filts == cf.end_filts*4:
            x=self.conv_shared5(x)
        elif filts == cf.end_filts*2:
            x=self.conv_shared6(x)
        elif filts == cf.end_filts:
            x=self.conv_shared7(x)
        # Shared convolutional base of the RPN.
        #print("x.shape",x.shape)
        # Anchor Score. (batch, anchors per location * 2, y, x, (z)).
        rpn_class_logits = self.conv_class(x)
        #print("rpn class shape",rpn_class_logits.shape)
        # Reshape to (batch, 2, anchors)
        axes = (0, 2, 3, 1) 
        rpn_class_logits = rpn_class_logits.permute(*axes)
        #print("rpn class logits shape",rpn_class_logits.shape)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)
        #print("rpn class logits shape after",rpn_class_logits.shape)
        # Softmax on last dimension (fg vs. bg).
        rpn_probs = F.softmax(rpn_class_logits, dim=1)
        #print(rpn_probs.shape)
        # Bounding box refinement. (batch, anchors_per_location * (y, x, (z), log(h), log(w), (log(d)), y, x, (z))
        rpn_bbox = self.conv_bbox(x)
        #print("rpb box shape",rpn_bbox.shape)
        # Reshape to (batch, 2*dim, anchors)
        rpn_bbox = rpn_bbox.permute(*axes)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 2* 2)
        #print("rpb box shape after",rpn_bbox.shape)
        return [rpn_class_logits, rpn_probs, rpn_bbox]

def pyramid_roi_align(feature_maps_orig, rois, pool_size, pyramid_levels, dim):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
    :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
    :param pool_size: list of poolsizes in dims: [x, y, (z)]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))
    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    feature_maps=[]
    for i in range(len(feature_maps_orig)):
      mid=int(feature_maps_orig[i].shape[2]/2)#2 means z coordinates
      feature_maps.append(feature_maps_orig[i][:,:,mid,:,:])
  
    boxes = rois[:, :dim*2]
    batch_ixs = rois[:, dim*2]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if dim == 2:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    else:
        y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)

    h = y2 - y1
    w = x2 - x1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + mutils.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
    # if Pyramid contains additional level P6, adapt the roi_level assignemnt accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h*w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []
    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_ixs[ix].int()

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        #
        # Here we use the simplified approach of a single value per bin,
        # which is how is done in tf.crop_and_resize()
        #
        # Also fixed a bug from original implementation, reported in:
        # https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35

        if len(pool_size) == 2:
            pooled_features = ra2D(pool_size[0], pool_size[1], 0)(feature_maps[level_ix], level_boxes, ind)
        else:
            pooled_features = ra3D(pool_size[0], pool_size[1], pool_size[2], 0)(feature_maps[level_ix], level_boxes, ind)

        pooled.append(pooled_features)


    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

class Classifier(nn.Module):
    """
    Head network for classification and bounding box refinement. Performs RoiAlign, processes resulting features through a
    shared convolutional base and finally branches off the classifier- and regression head.
    """
    def __init__(self, cf, conv):
        super(Classifier, self).__init__()
        self.dim = conv.dim
        self.in_channels = cf.end_filts
        self.pool_size = cf.pool_size
        self.pyramid_levels = cf.pyramid_levels
        # instance_norm does not work with spatial dims (1, 1, (1))
        norm = cf.norm if cf.norm != 'instance_norm' else None

        self.conv1 = conv(cf.end_filts, cf.end_filts * 4, kernel_size=self.pool_size, stride=1, norm=norm, relu=cf.relu)
        self.conv2 = conv(cf.end_filts * 4, cf.end_filts * 4, kernel_size=1, stride=1, norm=norm, relu=cf.relu)
        self.linear_class = nn.Linear(cf.end_filts * 4, cf.head_classes)
        self.linear_bbox = nn.Linear(cf.end_filts * 4, cf.head_classes * 2 * self.dim)
    def forward(self, x, rois):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :param rois: normalized box coordinates as proposed by the RPN to be forwarded through
        the second stage (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix). Proposals of all batch elements
        have been merged to one vector, while the origin info has been stored for re-allocation.
        :return: mrcnn_class_logits (n_proposals, n_head_classes)
        :return: mrcnn_bbox (n_proposals, n_head_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        """
        x = pyramid_roi_align(x, rois, self.pool_size, self.pyramid_levels, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_channels * 4)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, self.dim * 2)
        return [mrcnn_class_logits, mrcnn_bbox]

def refine_detections(rois, probs, deltas, batch_ixs, cf):
    """
    Refine classified proposals, filter overlaps and return final detections.
    :param rois: (n_proposals, 2 * dim) normalized boxes as proposed by RPN. n_proposals = batch_size * POST_NMS_ROIS
    :param probs: (n_proposals, n_classes) softmax probabilities for all rois as predicted by mrcnn classifier.
    :param deltas: (n_proposals, n_classes, 2 * dim) box refinement deltas as predicted by mrcnn bbox regressor.
    :param batch_ixs: (n_proposals) batch element assignemnt info for re-allocation.
    :return: result: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score))
    """
    # class IDs per ROI. Since scores of all classes are of interest (not just max class), all are kept at this point.
    class_ids = []
    fg_classes = cf.head_classes - 1
    # repeat vectors to fill in predictions for all foreground classes.
    for ii in range(1, fg_classes + 1):
        class_ids += [ii] * rois.shape[0]
    class_ids = torch.from_numpy(np.array(class_ids)).cuda()

    rois = rois.repeat(fg_classes, 1)
    probs = probs.repeat(fg_classes, 1)
    deltas = deltas.repeat(fg_classes, 1, 1)
    batch_ixs = batch_ixs.repeat(fg_classes)

    # get class-specific scores and  bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long().cuda()
    class_scores = probs[idx, class_ids]
    deltas_specific = deltas[idx, class_ids]
    batch_ixs = batch_ixs[idx]

    # apply bounding box deltas. re-scale to image coordinates.
    std_dev = torch.from_numpy(np.reshape(cf.rpn_bbox_std_dev, [1, cf.dim * 2])).float().cuda()
    scale = torch.from_numpy(cf.scale).float().cuda()
    refined_rois = mutils.apply_box_deltas_2D(rois, deltas_specific * std_dev) * scale if cf.dim == 2 else \
        mutils.apply_box_deltas_3D(rois, deltas_specific * std_dev) * scale

    # round and cast to int since we're deadling with pixels now
    refined_rois = mutils.clip_to_window(cf.window, refined_rois)
    refined_rois = torch.round(refined_rois)

    # filter out low confidence boxes
    keep = idx
    keep_bool = (class_scores >= cf.model_min_confidence)
    if 0 not in torch.nonzero(keep_bool).size():

        score_keep = torch.nonzero(keep_bool)[:, 0]
        pre_nms_class_ids = class_ids[score_keep]
        pre_nms_rois = refined_rois[score_keep]
        pre_nms_scores = class_scores[score_keep]
        pre_nms_batch_ixs = batch_ixs[score_keep]

        for j, b in enumerate(mutils.unique1d(pre_nms_batch_ixs)):

            bixs = torch.nonzero(pre_nms_batch_ixs == b)[:, 0]
            bix_class_ids = pre_nms_class_ids[bixs]
            bix_rois = pre_nms_rois[bixs]
            bix_scores = pre_nms_scores[bixs]

            for i, class_id in enumerate(mutils.unique1d(bix_class_ids)):

                ixs = torch.nonzero(bix_class_ids == class_id)[:, 0]
                # nms expects boxes sorted by score.
                ix_rois = bix_rois[ixs]
                ix_scores = bix_scores[ixs]
                ix_scores, order = ix_scores.sort(descending=True)
                ix_rois = ix_rois[order, :]

                if cf.dim == 2:
                    class_keep = nms_2D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)
                else:
                    class_keep = nms_3D(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1), cf.detection_nms_threshold)

                # map indices back.
                class_keep = keep[score_keep[bixs[ixs[order[class_keep]]]]]
                # merge indices over classes for current batch element
                b_keep = class_keep if i == 0 else mutils.unique1d(torch.cat((b_keep, class_keep)))

            # only keep top-k boxes of current batch-element
            top_ids = class_scores[b_keep].sort(descending=True)[1][:cf.model_max_instances_per_batch_element]
            b_keep = b_keep[top_ids]

            # merge indices over batch elements.
            batch_keep = b_keep if j == 0 else mutils.unique1d(torch.cat((batch_keep, b_keep)))

        keep = batch_keep

    else:
        keep = torch.tensor([0]).long().cuda()

    # arrange output
    result = torch.cat((refined_rois[keep],
                        batch_ixs[keep].unsqueeze(1),
                        class_ids[keep].unsqueeze(1).float(),
                        class_scores[keep].unsqueeze(1)), dim=1)

    return result


def compute_rpn_class_loss(rpn_match, rpn_class_logits, shem_poolsize):
    """
    :param rpn_match: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :param rpn_class_logits: (n_anchors, 2). logits from RPN classifier.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample
    (stochastic-hard-example-mining).
    :return: loss: torch tensor
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """

    # filter out neutral anchors.
    pos_indices = torch.nonzero(rpn_match == 1)
    neg_indices = torch.nonzero(rpn_match == -1)

    # loss for positive samples
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = rpn_class_logits[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, torch.LongTensor([1] * pos_indices.shape[0]).cuda())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # loss for negative samples: draw hard negative examples (SHEM)
    # that match the number of positive samples, but at least 1.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = rpn_class_logits[neg_indices]
        negative_count = np.max((1, pos_indices.cpu().data.numpy().size))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_rpn_bbox_loss(rpn_target_deltas, rpn_pred_deltas, rpn_match):
    """
    :param rpn_target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param rpn_pred_deltas: predicted deltas from RPN. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param rpn_match: (n_anchors). [-1, 0, 1] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(rpn_match == 1).size():

        indices = torch.nonzero(rpn_match == 1).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        rpn_pred_deltas = rpn_pred_deltas[indices]
        # Trim target bounding box deltas to the same length as rpn_bbox.
        target_deltas = rpn_target_deltas[:rpn_pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(rpn_pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """
    :param target_class_ids: (n_sampled_rois) batch dimension was merged into roi dimension.
    :param pred_class_logits: (n_sampled_rois, n_classes)
    :return: loss: torch 1D tensor.
    """
    if 0 not in target_class_ids.size():
        #print("compute mrcnn class loss",target_class_ids.long().size())
        #print("compute mrcnn class loss",pred_class_logits.size())
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
        """
        test=torch.tensor([1,1,1])
        test=test.cuda()
        loss_test = F.cross_entropy(pred_class_logits, test.long())
        print("compute mrcnn class loss loss_test",loss_test)
        """
    else:
        #print("compute mrcnn class loss has zero",target_class_ids.size())
        loss = torch.FloatTensor([0.]).cuda()

    return loss


def compute_mrcnn_bbox_loss(mrcnn_target_deltas, mrcnn_pred_deltas, target_class_ids):
    """
    :param mrcnn_target_deltas: (n_sampled_rois, (dy, dx, (dz), log(dh), log(dw), (log(dh)))
    :param mrcnn_pred_deltas: (n_sampled_rois, n_classes, (dy, dx, (dz), log(dh), log(dw), (log(dh)))
    :param target_class_ids: (n_sampled_rois)
    :return: loss: torch 1D tensor.
    """
    
    if 0 not in torch.nonzero(target_class_ids > 0).size():
        
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix].long()
        target_bbox = mrcnn_target_deltas[positive_roi_ix, :].detach()
        pred_bbox = mrcnn_pred_deltas[positive_roi_ix, positive_roi_class_ids, :]
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, pred_masks, target_class_ids):
    """
    :param target_masks: (n_sampled_rois, y, x, (z)) A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    :param pred_masks: (n_sampled_rois, n_classes, y, x, (z)) float32 tensor with values between [0, 1].
    :param target_class_ids: (n_sampled_rois)
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(target_class_ids > 0).size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix].long()
        y_true = target_masks[positive_ix, :, :].detach()
        y_pred = pred_masks[positive_ix, positive_class_ids, :, :]
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

def loss_samples_forward(self, batch_gt_class_ids, batch_gt_boxes):
        """
        this is the second forward pass through the second stage (features from stage one are re-used).
        samples few rois in detection_target_layer and forwards only those for loss computation.
        :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
        :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
        :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, y, x, (z), c)
        :return: sample_logits: (n_sampled_rois, n_classes) predicted class scores.
        :return: sample_boxes: (n_sampled_rois, n_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        :return: sample_mask: (n_sampled_rois, n_classes, y, x, (z)) predicted masks per class and proposal.
        :return: sample_target_class_ids: (n_sampled_rois) target class labels of sampled proposals.
        :return: sample_target_deltas: (n_sampled_rois, 2 * dim) target deltas of sampled proposals for box refinement.
        :return: sample_target_masks: (n_sampled_rois, y, x, (z)) target masks of sampled proposals.
        :return: sample_proposals: (n_sampled_rois, 2 * dim) RPN output for sampled proposals. only for monitoring/plotting.
        """
        # sample rois for loss and get corresponding targets for all Mask R-CNN head network losses.
        sample_ix, sample_target_class_ids, sample_target_deltas, sample_target_mask = \
            detection_target_layer(self.rpn_rois_batch_info, self.batch_mrcnn_class_scores,
                                   batch_gt_class_ids, batch_gt_boxes,self.cf)

        # re-use feature maps and RPN output from first forward pass.
        sample_proposals = self.rpn_rois_batch_info[sample_ix]
        if 0 not in sample_proposals.size():
            sample_logits, sample_boxes = self.classifier(self.mrcnn_feature_maps, sample_proposals)
            #sample_mask = self.mask(self.mrcnn_feature_maps, sample_proposals)
        else:
            sample_logits = torch.FloatTensor().cuda()
            sample_boxes = torch.FloatTensor().cuda()
            #sample_mask = torch.FloatTensor().cuda()

        return [sample_logits, sample_boxes, sample_mask, sample_target_class_ids, sample_target_deltas,
                sample_target_mask, sample_proposals]

def detection_target_layer(batch_proposals, batch_mrcnn_class_scores, batch_gt_class_ids, batch_gt_boxes, cf):
    """
    Subsamples proposals for mrcnn losses and generates targets. Sampling is done per batch element, seems to have positive
    effects on training, as opposed to sampling over entire batch. Negatives are sampled via stochastic-hard-example-mining
    (SHEM), where a number of negative proposals are drawn from larger pool of highest scoring proposals for stochasticity.
    Scoring is obtained here as the max over all foreground probabilities as returned by mrcnn_classifier (worked better than
    loss-based class balancing methods like "online-hard-example-mining" or "focal loss".)
    :param batch_proposals: (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs).
    boxes as proposed by RPN. n_proposals here is determined by batch_size * POST_NMS_ROIS.
    :param batch_mrcnn_class_scores: (n_proposals, n_classes)
    :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
    :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
    :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, y, x, (z), c)
    :return: sample_indices: (n_sampled_rois) indices of sampled proposals to be used for loss functions.
    :return: target_class_ids: (n_sampled_rois)containing target class labels of sampled proposals.
    :return: target_deltas: (n_sampled_rois, 2 * dim) containing target deltas of sampled proposals for box refinement.
    :return: target_masks: (n_sampled_rois, y, x, (z)) containing target masks of sampled proposals.
    """
    # normalization of target coordinates
    if cf.dim == 2:
        h, w = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w])).float().cuda()
    else:
        h, w, z = cf.patch_size
        scale = torch.from_numpy(np.array([h, w, h, w, z, z])).float().cuda()

   
    positive_count = 0
    negative_count = 0
    sample_positive_indices = []
    sample_negative_indices = []
    sample_deltas = []
    sample_masks = []
    sample_class_ids = []

    # loop over batch and get positive and negative sample rois.
    for b in range(len(batch_gt_class_ids)):
        #print("detection_target_layer gt_class_ids",gt_class_ids)
        gt_class_ids = torch.from_numpy(batch_gt_class_ids[b]).int().cuda()
        
        #gt_masks = torch.from_numpy(batch_gt_masks[b]).float().cuda()
        #print("batch_gt_boxes[b]",batch_gt_boxes[b])
        if np.any(batch_gt_class_ids[b] > 0):  # skip roi selection for no gt images.
            gt_boxes = torch.tensor(batch_gt_boxes[b]).float().cuda() / scale
        else:
            gt_boxes = torch.FloatTensor().cuda()

        # get proposals and indices of current batch element.
        #print("batch_proposals",batch_proposals.shape)
        
        proposals = batch_proposals[batch_proposals[:, -1] == b][:, :-1]
        #print("proposals",proposals.shape)
        batch_element_indices = torch.nonzero(batch_proposals[:, -1] == b).squeeze(1)

        # Compute overlaps matrix [proposals, gt_boxes]
        #print("gt_boxes.shape",gt_boxes.shape)
        gt_boxes=gt_boxes.view(1,-1)
        #print("detec target gt_boxes",gt_boxes.shape)
        if 0 not in gt_boxes.size():
            if gt_boxes.shape[1] == 4:
                overlaps = mutils.bbox_overlaps_2D(proposals, gt_boxes)
            else:
                overlaps = mutils.bbox_overlaps_3D(proposals, gt_boxes)
            #print("detection_target_layer overlaps",overlaps)
           
            # Determine postive and negative ROIs
            roi_iou_max = torch.max(overlaps, dim=1)[0]
            #print("detection_target_layer roi_iou_max",roi_iou_max)
            # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
            positive_roi_bool = roi_iou_max >= (0.2 if cf.dim == 2 else 0.3)
            #print("detection_target_layer positive_roi_bool",positive_roi_bool)
           
            # 2. Negative ROIs are those with < 0.1 with every GT box.
            negative_roi_bool = roi_iou_max < (0.1 if cf.dim == 2 else 0.01)
        else:
            positive_roi_bool = torch.FloatTensor().cuda()
            negative_roi_bool = torch.from_numpy(np.array([1]*proposals.shape[0])).cuda()

        # Sample Positive ROIs
        if 0 not in torch.nonzero(positive_roi_bool).size():
            positive_indices = torch.nonzero(positive_roi_bool).squeeze(1)
           
            #print("detection_target_layer positive_indices",positive_indices)
            positive_samples = int(cf.train_rois_per_image * cf.roi_positive_ratio)
            rand_idx = torch.randperm(positive_indices.size()[0])
            
            rand_idx = rand_idx[:positive_samples].cuda()
            
            positive_indices = positive_indices[rand_idx]
            
            positive_samples = positive_indices.size()[0]
            positive_rois = proposals[positive_indices, :]
            # Assign positive ROIs to GT boxes.
            positive_overlaps = overlaps[positive_indices, :]
            
            roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
           
            roi_gt_boxes = gt_boxes[roi_gt_box_assignment, :]
            roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment]
           
            # Compute bbox refinement targets for positive ROIs
            deltas = mutils.box_refinement(positive_rois, roi_gt_boxes)
            std_dev = torch.from_numpy(cf.bbox_std_dev).float().cuda()
            deltas /= std_dev

            # Assign positive ROIs to GT masks
            #roi_masks = gt_masks[roi_gt_box_assignment, :, :]

            # Compute mask targets
            boxes = positive_rois
            #box_ids = torch.arange(roi_masks.size()[0]).int().cuda()

            # if len(cf.mask_shape) == 2:
                # masks = ra2D(cf.mask_shape[0], cf.mask_shape[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids)
            # else:
                # masks = ra3D(cf.mask_shape[0], cf.mask_shape[1], cf.mask_shape[2], 0)(roi_masks.unsqueeze(1), boxes, box_ids)

            # masks = masks.squeeze(1)
            # # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
            # # binary cross entropy loss.
            # masks = torch.round(masks)

            sample_positive_indices.append(batch_element_indices[positive_indices])
            sample_deltas.append(deltas)
            #sample_masks.append(masks)
            sample_class_ids.append(roi_gt_class_ids)
            
            positive_count += positive_samples
        else:
            positive_samples = 0

        # Negative ROIs. Add enough to maintain positive:negative ratio, but at least 1. Sample via SHEM.
        if 0 not in torch.nonzero(negative_roi_bool).size():
            negative_indices = torch.nonzero(negative_roi_bool).squeeze(1)
            r = 1.0 / cf.roi_positive_ratio
            b_neg_count = np.max((int(r * positive_samples - positive_samples), 1))
            roi_probs_neg = batch_mrcnn_class_scores[batch_element_indices[negative_indices]]
            raw_sampled_indices = mutils.shem(roi_probs_neg, b_neg_count, cf.shem_poolsize)
            sample_negative_indices.append(batch_element_indices[negative_indices[raw_sampled_indices]])
            negative_count += raw_sampled_indices.size()[0]

    if len(sample_positive_indices) > 0:
        target_deltas = torch.cat(sample_deltas)
        #target_masks = torch.cat(sample_masks)
        target_class_ids = torch.cat(sample_class_ids)

    # Pad target information with zeros for negative ROIs.
    #print("detection_target_layer positive_count",positive_count)
    #print("detection_target_layer negative_count",negative_count)
    if positive_count > 0 and negative_count > 0:
        sample_indices = torch.cat((torch.cat(sample_positive_indices), torch.cat(sample_negative_indices)), dim=0)
        zeros = torch.zeros(negative_count).int().cuda()
        target_class_ids = torch.cat([target_class_ids, zeros], dim=0)
        zeros = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_deltas = torch.cat([target_deltas, zeros], dim=0)
        zeros = torch.zeros(negative_count, *cf.mask_shape).cuda()
        #target_masks = torch.cat([target_masks, zeros], dim=0)
    elif positive_count > 0:
        sample_indices = torch.cat(sample_positive_indices)
    elif negative_count > 0:
        sample_indices = torch.cat(sample_negative_indices)
        zeros = torch.zeros(negative_count).int().cuda()
        target_class_ids = zeros
        zeros = torch.zeros(negative_count, cf.dim * 2).cuda()
        target_deltas = zeros
        zeros = torch.zeros(negative_count, *cf.mask_shape).cuda()
        #target_masks = zeros
    else:
        sample_indices = torch.LongTensor().cuda()
        target_class_ids = torch.IntTensor().cuda()
        target_deltas = torch.FloatTensor().cuda()
        #target_masks = torch.FloatTensor().cuda()

    return sample_indices, target_class_ids, target_deltas, #target_masks

def get_results(cf, img_shape, detections, box_results_list=None, return_masks=False):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :param return_masks: boolean. If True, full resolution masks are returned for all proposals (speed trade-off).
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, 1] only fg. vs. bg for now.
             class-specific return of masks will come with implementation of instance segmentation evaluation.
    """
    detections = detections.cpu().data.numpy()
    #print("detections.shape",detections.shape)
    # if cf.dim == 2:
        # detection_masks = detection_masks.permute(0, 2, 3, 1).cpu().data.numpy()
    # else:
        # detection_masks = detection_masks.permute(0, 2, 3, 4, 1).cpu().data.numpy()

    # restore batch dimension of merged detections using the batch_ix info.
    batch_ixs = detections[:, cf.dim*2]
    detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]
    #print("after detections.shape",detections.shape)
    #mrcnn_mask = [detection_masks[batch_ixs == ix] for ix in range(img_shape[0])]

    # for test_forward, where no previous list exists.
    if box_results_list is None:
        box_results_list = [[] for _ in range(img_shape[0])]

    seg_preds = []
    # loop over batch and unmold detections.
    for ix in range(img_shape[0]):

        if 0 not in detections[ix].shape:
            boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
            class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
            scores = detections[ix][:, 2 * cf.dim + 2]
            #masks = mrcnn_mask[ix][np.arange(boxes.shape[0]), ..., class_ids]

            # Filter out detections with zero area. Often only happens in early
            # stages of training when the network weights are still a bit random.
            if cf.dim == 2:
                exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
            else:
                exclude_ix = np.where(
                    (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]

            if exclude_ix.shape[0] > 0:
                boxes = np.delete(boxes, exclude_ix, axis=0)
                class_ids = np.delete(class_ids, exclude_ix, axis=0)
                scores = np.delete(scores, exclude_ix, axis=0)
                #masks = np.delete(masks, exclude_ix, axis=0)

            # Resize masks to original image size and set boundary threshold.
            full_masks = []
            permuted_image_shape = list(img_shape[2:]) + [img_shape[1]]
            if return_masks:
                for i in range(masks.shape[0]):
                    # Convert neural network mask to full size mask.
                    full_masks.append(mutils.unmold_mask_2D(masks[i], boxes[i], permuted_image_shape)
                    if cf.dim == 2 else mutils.unmold_mask_3D(masks[i], boxes[i], permuted_image_shape))
            # if masks are returned, take max over binary full masks of all predictions in this image.
            # right now only binary masks for plotting/monitoring. for instance segmentation return all proposal maks.
            final_masks = np.max(np.array(full_masks), 0) if len(full_masks) > 0 else np.zeros(
                (*permuted_image_shape[:-1],))

            # add final perdictions to results.
            if 0 not in boxes.shape:
                for ix2, score in enumerate(scores):
                    box_results_list[ix].append({'box_coords': boxes[ix2], 'box_score': score,
                                                 'box_type': 'det', 'box_pred_class_id': class_ids[ix2]})
        else:
            # pad with zero dummy masks.
            final_masks = np.zeros(img_shape[2:])

        seg_preds.append(final_masks)

    # create and fill results dictionary.
    results_dict = {'boxes': box_results_list,
                    'seg_preds': np.round(np.array(seg_preds))[:, np.newaxis].astype('uint8')}

    return results_dict




class net(nn.Module):


    def __init__(self, cf):

        super(net, self).__init__()
        self.cf = cf
        #self.logger = logger
        self.build()

        if self.cf.weight_init is not None:
            #logger.info("using pytorch weight init of type {}".format(self.cf.weight_init))
            mutils.initialize_weights(self)
        #else:
         #   logger.info("using default pytorch weight init")


    def build(self):
        """Build Mask R-CNN architecture."""

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2**5 != int(h / 2**5) or w / 2**5 != int(w / 2**5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        if len(self.cf.patch_size) == 3:
            d = self.cf.patch_size[2]
            if d / 2**3 != int(d / 2**3):
                raise Exception("Image z dimension must be dividable by 2 at least 3 times "
                                "to avoid fractions when downscaling and upscaling.")
        self.fpn=FPN(cf,nn.Conv3d).cuda()
        self.conv=NDConv(3)
        self.conv2=NDConv(2)
        self.rpn=RPN(cf,self.conv,self.conv2).cuda()
        self.classifier = Classifier(self.cf, self.conv2).cuda()

        # instanciate abstract multi dimensional conv class and backbone class.
        # conv = mutils.NDConvGenerator(self.cf.dim)
        # backbone = utils.import_module('bbone', self.cf.backbone_path)

        # build Anchors, FPN, RPN, Classifier / Bbox-Regressor -head, Mask-head
        self.np_anchors = mutils.generate_pyramid_anchors( self.cf)
        self.anchors = torch.from_numpy(self.np_anchors).float().cuda()


    def loss_samples_forward(self, batch_gt_class_ids, batch_gt_boxes):
        """
        this is the second forward pass through the second stage (features from stage one are re-used).
        samples few rois in detection_target_layer and forwards only those for loss computation.
        :param batch_gt_class_ids: list over batch elements. Each element is a list over the corresponding roi target labels.
        :param batch_gt_boxes: list over batch elements. Each element is a list over the corresponding roi target coordinates.
        :param batch_gt_masks: list over batch elements. Each element is binary mask of shape (n_gt_rois, y, x, (z), c)
        :return: sample_logits: (n_sampled_rois, n_classes) predicted class scores.
        :return: sample_boxes: (n_sampled_rois, n_classes, 2 * dim) predicted corrections to be applied to proposals for refinement.
        :return: sample_mask: (n_sampled_rois, n_classes, y, x, (z)) predicted masks per class and proposal.
        :return: sample_target_class_ids: (n_sampled_rois) target class labels of sampled proposals.
        :return: sample_target_deltas: (n_sampled_rois, 2 * dim) target deltas of sampled proposals for box refinement.
        :return: sample_target_masks: (n_sampled_rois, y, x, (z)) target masks of sampled proposals.
        :return: sample_proposals: (n_sampled_rois, 2 * dim) RPN output for sampled proposals. only for monitoring/plotting.
        """
        # sample rois for loss and get corresponding targets for all Mask R-CNN head network losses.
        sample_ix, sample_target_class_ids, sample_target_deltas= detection_target_layer(self.rpn_rois_batch_info, self.batch_mrcnn_class_scores, batch_gt_class_ids, batch_gt_boxes,  self.cf)

        # re-use feature maps and RPN output from first forward pass.
        sample_proposals = self.rpn_rois_batch_info[sample_ix]
        if 0 not in sample_proposals.size():
            sample_logits, sample_boxes = self.classifier(self.mrcnn_feature_maps, sample_proposals)
            #sample_mask = self.mask(self.mrcnn_feature_maps, sample_proposals)
        else:
            sample_logits = torch.FloatTensor().cuda()
            sample_boxes = torch.FloatTensor().cuda()
            #sample_mask = torch.FloatTensor().cuda()

        return [sample_logits, sample_boxes, sample_target_class_ids, sample_target_deltas, sample_proposals]



    def train_forward(self, batch, is_validation=False):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network. prepares input data
        for processing, computes losses, and stores outputs in a dictionary.
        :param batch: dictionary containing 'data', 'seg', etc.
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
                'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes].
                'monitor_values': dict of values to be monitored.
        """
        img = batch['data']
        gt_class_ids = batch['roi_labels']
        #print("forward gt_class_ids ",gt_class_ids)
        gt_boxes = batch['bb_target']
        gt_boxes=gt_boxes.reshape(img.shape[0],1,-1)
        #print("train gt_boxes shape",gt_boxes.shape)
        axes = (0, 2, 3, 1) if self.cf.dim == 2 else (0, 2, 3, 4, 1)
        #gt_masks = [np.transpose(batch['roi_masks'][ii], axes=axes) for ii in range(len(batch['roi_masks']))]

        
        img = torch.tensor(img).float().cuda()
        #print("img shape",img.shape)
        batch_rpn_class_loss = torch.FloatTensor([0]).cuda()
        batch_rpn_bbox_loss = torch.FloatTensor([0]).cuda()

        # list of output boxes for monitoring/plotting. each element is a list of boxes per batch element.
        box_results_list = [[] for _ in range(img.shape[0])]

        #forward passes. 1. general forward pass, where no activations are saved in second stage (for performance
        # monitoring and loss sampling). 2. second stage forward pass of sampled rois with stored activations for backprop.
        
        #rpn_class_logits, rpn_pred_deltas, proposal_boxes, detections, detection_masks = self.forward(img)
        rpn_class_logits, rpn_pred_deltas, proposal_boxes, detections = self.forward(img)
        #print("train for detection shape",detections.shape)
        mrcnn_class_logits, mrcnn_pred_deltas, target_class_ids, mrcnn_target_deltas,  \
        sample_proposals = self.loss_samples_forward(gt_class_ids, gt_boxes)

        # loop over batch
        for b in range(img.shape[0]):
            if len(gt_boxes[b]) > 0:

                # add gt boxes to output list for monitoring.
                for ix in range(len(gt_boxes[b])):
                    box_results_list[b].append({'box_coords': batch['bb_target'][b][ix],
                                                'box_label': batch['roi_labels'][b][ix], 'box_type': 'gt'})

                # match gt boxes with anchors to generate targets for RPN losses.
                #print("type(self.np_anchors)",type(self.np_anchors))
                #print("type(gt_boxes[b])",type(gt_boxes[b]))
                rpn_match, rpn_target_deltas = mutils.gt_anchor_matching(self.cf, self.np_anchors, gt_boxes[b])

                # add positive anchors used for loss to output list for monitoring.
                pos_anchors = mutils.clip_boxes_numpy(self.np_anchors[np.argwhere(rpn_match == 1)][:, 0], img.shape[2:])
                for p in pos_anchors:
                    box_results_list[b].append({'box_coords': p, 'box_type': 'pos_anchor'})

            else:
                rpn_match = np.array([-1]*self.np_anchors.shape[0])
                rpn_target_deltas = np.array([0])

            rpn_match = torch.from_numpy(rpn_match).cuda()
            rpn_target_deltas = torch.from_numpy(rpn_target_deltas).float().cuda()

            # compute RPN losses.
            rpn_class_loss, neg_anchor_ix = compute_rpn_class_loss(rpn_match, rpn_class_logits[b], self.cf.shem_poolsize)
            rpn_bbox_loss = compute_rpn_bbox_loss(rpn_target_deltas, rpn_pred_deltas[b], rpn_match)
            batch_rpn_class_loss += rpn_class_loss / img.shape[0]
            batch_rpn_bbox_loss += rpn_bbox_loss / img.shape[0]

            # add negative anchors used for loss to output list for monitoring.
            neg_anchors = mutils.clip_boxes_numpy(self.np_anchors[np.argwhere(rpn_match == -1)][0, neg_anchor_ix], img.shape[2:])
            for n in neg_anchors:
                box_results_list[b].append({'box_coords': n, 'box_type': 'neg_anchor'})

            # add highest scoring proposals to output list for monitoring.
            rpn_proposals = proposal_boxes[b][proposal_boxes[b, :, -1].argsort()][::-1]
            for r in rpn_proposals[:self.cf.n_plot_rpn_props, :-1]:
                box_results_list[b].append({'box_coords': r, 'box_type': 'prop'})

        # add positive and negative roi samples used for mrcnn losses to output list for monitoring.
        if 0 not in sample_proposals.shape:
            rois = mutils.clip_to_window(self.cf.window, sample_proposals).cpu().data.numpy()
            for ix, r in enumerate(rois):
                box_results_list[int(r[-1])].append({'box_coords': r[:-1] * self.cf.scale,
                                            'box_type': 'pos_class' if target_class_ids[ix] > 0 else 'neg_class'})

        batch_rpn_class_loss = batch_rpn_class_loss
        batch_rpn_bbox_loss = batch_rpn_bbox_loss

        # compute mrcnn losses.
        #print("train forward target_class_ids",target_class_ids)
        #print("train forward mrcnn_class_logits",mrcnn_class_logits)
        mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
        mrcnn_bbox_loss = compute_mrcnn_bbox_loss(mrcnn_target_deltas, mrcnn_pred_deltas, target_class_ids)
        #print("train forward mrcnn_bbox_loss",mrcnn_bbox_loss)
        #print("train forward mrcnn_class_loss",mrcnn_class_loss)

        # mrcnn can be run without pixelwise annotations available (Faster R-CNN mode).
        # In this case, the mask_loss is taken out of training.
        # if not self.cf.frcnn_mode:
            # mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, mrcnn_pred_mask, target_class_ids)
        # else:
            # mrcnn_mask_loss = torch.FloatTensor([0]).cuda()

        loss = batch_rpn_class_loss + batch_rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss #+ mrcnn_mask_loss

        # monitor RPN performance: detection count = the number of correctly matched proposals per fg-class.
        dcount = [list(target_class_ids.cpu().data.numpy()).count(c) for c in np.arange(self.cf.head_classes)[1:]]



        # run unmolding of predictions for monitoring and merge all results to one dictionary.
        #return_masks = self.cf.return_masks_in_val if is_validation else False
        results_dict = get_results(self.cf, img.shape, detections, 
                                   box_results_list)

        results_dict['torch_loss'] = loss
        results_dict['batch_rpn_class_loss'] = batch_rpn_class_loss 
        results_dict['batch_rpn_bbox_loss'] = batch_rpn_bbox_loss
        results_dict['mrcnn_bbox_loss'] = mrcnn_bbox_loss
        results_dict['mrcnn_class_loss'] = mrcnn_class_loss
        results_dict['monitor_values'] = {'loss': loss.item(), 'class_loss': mrcnn_class_loss.item()}

        # results_dict['logger_string'] =  \
            # "loss: {0:.2f}, rpn_class: {1:.2f}, rpn_bbox: {2:.2f}, mrcnn_class: {3:.2f}, mrcnn_bbox: {4:.2f}, " \
            # "mrcnn_mask: {5:.2f}, dcount {6}".format(loss.item(), batch_rpn_class_loss.item(),
                                                     # batch_rpn_bbox_loss.item(), mrcnn_class_loss.item(),
                                                     # mrcnn_bbox_loss.item(),  dcount)

        return results_dict

    def test_forward(self, batch, return_masks=True):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :param return_masks: boolean. If True, full resolution masks are returned for all proposals (speed trade-off).
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, n_classes]
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        _, _, _, detections = self.forward(img)
        results_dict = get_results(self.cf, img.shape, detections, return_masks=return_masks)
        return results_dict

    def forward(self, img, is_training=True):
        """
        :param img: input images (b, c, y, x, (z)).
        :return: rpn_pred_logits: (b, n_anchors, 2)
        :return: rpn_pred_deltas: (b, n_anchors, (y, x, (z), log(h), log(w), (log(d))))
        :return: batch_proposal_boxes: (b, n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix)) only for monitoring/plotting.
        :return: detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
        #:return: detection_masks: (n_final_detections, n_classes, y, x, (z)) raw molded masks as returned by mask-head.
        """
        # extract features.
        fpn_outs = self.fpn(img)

        rpn_feature_maps = [fpn_outs[i] for i in [0, 1, 2, 3]]
        """
        print(len(fpn_outs))
        print(fpn_outs[0].shape)
        print(fpn_outs[1].shape)
        print(fpn_outs[2].shape)
        print(fpn_outs[3].shape)
        """
        mrcnn_feature_maps = rpn_feature_maps
        self.mrcnn_feature_maps=mrcnn_feature_maps
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))
        """
        print("rpn")
        print()
        print(len(layer_outputs))
        
        for i in range(4):
          for j in range(3):
            print(layer_outputs[i][j].shape)
        
        """
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_pred_logits, rpn_pred_probs, rpn_pred_deltas = outputs


        is_training=True
        # generate proposals: apply predicted deltas to anchors and filter by foreground scores from RPN classifier.
        proposal_count = cf.post_nms_rois_training if is_training else cf.post_nms_rois_inference
        """
        print()
        print(rpn_pred_probs.shape)
        print(rpn_pred_deltas.shape)
        print(self.anchors.shape)
        print(proposal_count)
        """
        batch_rpn_rois, batch_proposal_boxes = proposal_layer(rpn_pred_probs, rpn_pred_deltas, proposal_count, self.anchors, self.cf)

        # merge batch dimension of proposals while storing allocation info in coordinate dimension.
        # batch_ixs = torch.from_numpy(np.repeat(np.arange(batch_rpn_rois.shape[0]), batch_rpn_rois.shape[1])).float().cuda()
        # rpn_rois = batch_rpn_rois.view(-1, batch_rpn_rois.shape[2])
        # rpn_rois_batch_info = torch.cat((rpn_rois, batch_ixs.unsqueeze(1)), dim=1)

        # this is the first of two forward passes in the second stage, where no activations are stored for backprop.
        # here, all proposals are forwarded (with virtual_batch_size = batch_size * post_nms_rois.)
        # for inference/monitoring as well as sampling of rois for the loss functions.
        # processed in chunks of roi_chunk_size to re-adjust to gpu-memory.



# chunked_rpn_rois = rpn_rois_batch_info.split(cf.roi_chunk_size)
# class_logits_list, bboxes_list = [], []
# with torch.no_grad():
            # for chunk in chunked_rpn_rois:
                # chunk_class_logits, chunk_bboxes = classifier(mrcnn_feature_maps, chunk)
                # class_logits_list.append(chunk_class_logits)
                # bboxes_list.append(chunk_bboxes)
# batch_mrcnn_class_logits = torch.cat(class_logits_list, 0)
# batch_mrcnn_bbox = torch.cat(bboxes_list, 0)
# batch_mrcnn_class_scores = F.softmax(batch_mrcnn_class_logits, dim=1)

        # # refine classified proposals, filter and return final detections.
# detections = refine_detections(rpn_rois, batch_mrcnn_class_scores, batch_mrcnn_bbox, batch_ixs, cf, )

        # # forward remaining detections through mask-head to generate corresponding masks.
# scale = [img.shape[2]] * 4 + [img.shape[-1]] * 2
# scale = torch.from_numpy(np.array(scale[:cf.dim * 2] + [1])[None]).float().cuda()


# detection_boxes = detections[:, :cf.dim * 2 + 1] / scale


        # fpn_outs = self.fpn(img)
        # rpn_feature_maps = [fpn_outs[i] for i in self.cf.pyramid_levels]
        # self.mrcnn_feature_maps = rpn_feature_maps

        # # loop through pyramid layers and apply RPN.
        # layer_outputs = []  # list of lists
        # for p in rpn_feature_maps:
            # layer_outputs.append(self.rpn(p))

        # # concatenate layer outputs.
        # # convert from list of lists of level outputs to list of lists of outputs across levels.
        # # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        # outputs = list(zip(*layer_outputs))
        # outputs = [torch.cat(list(o), dim=1) for o in outputs]
        # rpn_pred_logits, rpn_pred_probs, rpn_pred_deltas = outputs

        # # generate proposals: apply predicted deltas to anchors and filter by foreground scores from RPN classifier.
        # proposal_count = self.cf.post_nms_rois_training if is_training else self.cf.post_nms_rois_inference
        # batch_rpn_rois, batch_proposal_boxes = proposal_layer(rpn_pred_probs, rpn_pred_deltas, proposal_count, self.anchors, self.cf)

        # merge batch dimension of proposals while storing allocation info in coordinate dimension.
        batch_ixs = torch.from_numpy(np.repeat(np.arange(batch_rpn_rois.shape[0]), batch_rpn_rois.shape[1])).float().cuda()
        rpn_rois = batch_rpn_rois.view(-1, batch_rpn_rois.shape[2])
        self.rpn_rois_batch_info = torch.cat((rpn_rois, batch_ixs.unsqueeze(1)), dim=1)

        # this is the first of two forward passes in the second stage, where no activations are stored for backprop.
        # here, all proposals are forwarded (with virtual_batch_size = batch_size * post_nms_rois.)
        # for inference/monitoring as well as sampling of rois for the loss functions.
        # processed in chunks of roi_chunk_size to re-adjust to gpu-memory.
        chunked_rpn_rois = self.rpn_rois_batch_info.split(self.cf.roi_chunk_size)
        class_logits_list, bboxes_list = [], []
        with torch.no_grad():
            for chunk in chunked_rpn_rois:
                chunk_class_logits, chunk_bboxes = self.classifier(self.mrcnn_feature_maps, chunk)
                class_logits_list.append(chunk_class_logits)
                bboxes_list.append(chunk_bboxes)
        batch_mrcnn_class_logits = torch.cat(class_logits_list, 0)
        batch_mrcnn_bbox = torch.cat(bboxes_list, 0)
        self.batch_mrcnn_class_scores = F.softmax(batch_mrcnn_class_logits, dim=1)

        # refine classified proposals, filter and return final detections.
        detections = refine_detections(rpn_rois, self.batch_mrcnn_class_scores, batch_mrcnn_bbox, batch_ixs, self.cf, )

        # forward remaining detections through mask-head to generate corresponding masks.
        scale = [img.shape[2]] * 4 + [img.shape[-1]] * 2
        scale = torch.from_numpy(np.array(scale[:self.cf.dim * 2] + [1])[None]).float().cuda()

        #print("scale",scale)
        detection_boxes = detections[:, :self.cf.dim * 2 + 1] / scale
        # with torch.no_grad():
            # detection_masks = self.mask(self.mrcnn_feature_maps, detection_boxes)

        return [rpn_pred_logits, rpn_pred_deltas, batch_proposal_boxes, detections]


def IOU( box1, box2 ):
    box1=list(map(float, box1))
    box2=list(map(float, box2))
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0],box1[2],box2[0],box2[2])
    y_max = max(box1[1],box1[3],box2[1],box2[3])
    x_min = min(box1[0],box1[2],box2[0],box2[2])
    y_min = min(box1[1],box1[3],box2[1],box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio





batch={}
"""
mrcnn=net(cf)
batch['data']=np.random.randn(1,1,16,512,512)
batch['roi_labels']=np.array([[0,0,1]])
batch['bb_target']=np.array([[112,222,150,260]])
results_dict=mrcnn.train_forward(batch)
"""
print("first batch done")

mrcnn=net(cf).cuda()
optimizer = torch.optim.Adam(mrcnn.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
batch={}
#from sklearn.model_selection import train_test_split
#low_path_splits_train, low_path_splits_test, high_path_splits_train, high_path_splits_test= train_test_split( low_paths, high_paths, test_size=0.1, random_state=1)

dataset=dataset_3d()
data_loader = torch.utils.data.dataloader.DataLoader(dataset, cf.batch_size, shuffle=False)


for epoch in range(cf.num_epochs):
  torch.save(mrcnn.state_dict(),os.path.join(parm.out_dir,str(epoch)+ '_model_params.pkl'))
  
  #For each batch in the dataloader
  ave_batch_rpn_class_loss=0
  ave_batch_rpn_bbox_loss=0
  ave_mrcnn_class_loss=0
  ave_mrcnn_bbox_loss=0
  n_batch=1
  loss_total=[]
  for i, data in enumerate(data_loader, 0):
      image,bb_target,upper_place,lower_place,label=data
      #print("batch label",label)
      #print("image mean",torch.mean(image))
      #print("image var",torch.var(image))
      batch['data']=image
      #print("cf.num_epochs",cf.num_epochs)
      #print("cf.head_classes",cf.head_classes)
      #print("label",label)
      roi_labels=np.zeros((cf.batch_size,cf.head_classes))
      for j in range(cf.batch_size):
          #print("label[j]",label[j])
          roi_labels[j,int(label[j])]=1
      batch['roi_labels']=roi_labels
      batch['bb_target']=np.array(bb_target)
      #print("roi label",roi_labels)
      results_dict=mrcnn.train_forward(batch)
      batch_rpn_class_loss=results_dict['batch_rpn_class_loss'] 
      batch_rpn_bbox_loss=results_dict['batch_rpn_bbox_loss']
      mrcnn_class_loss=results_dict['mrcnn_class_loss']
      mrcnn_bbox_loss=results_dict['mrcnn_bbox_loss']
      """
      print("batch loss 1",batch_rpn_class_loss)
      print("batch loss 2",batch_rpn_bbox_loss)
      print("batch loss 3",mrcnn_class_loss)
      print("batch loss 4",mrcnn_bbox_loss)
      """
      ave_batch_rpn_class_loss=(ave_batch_rpn_class_loss*(n_batch-1)+batch_rpn_class_loss)*n_batch
      ave_batch_rpn_bbox_loss=(ave_batch_rpn_bbox_loss*(n_batch-1)+batch_rpn_bbox_loss)*n_batch
      ave_mrcnn_class_loss=(ave_mrcnn_class_loss*(n_batch-1)+mrcnn_class_loss)*n_batch
      ave_mrcnn_bbox_loss=(ave_mrcnn_bbox_loss*(n_batch-1)+mrcnn_bbox_loss)*n_batch
      
      
      #mrcnn.zero_grad()
      optimizer.zero_grad()
      results_dict['torch_loss'].backward()
      optimizer.step()
  print("batch_rpn_class_loss",ave_batch_rpn_class_loss)
  print("batch_rpn_bbox_loss",ave_batch_rpn_bbox_loss)
  print("mrcnn_class_loss",ave_mrcnn_class_loss)
  print("mrcnn_bbox_loss",ave_mrcnn_bbox_loss)
  loss_total.append([ave_batch_rpn_class_loss,ave_batch_rpn_bbox_loss,ave_mrcnn_class_loss,ave_mrcnn_bbox_loss])
  txt=open(os.path.join(parm.out_dir,'loss_total.txt'),'w')
  txt.write(str(loss_total))
  txt.close()
  