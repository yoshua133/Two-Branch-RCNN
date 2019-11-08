# Two-Branch-RCNN
This is a project to utilize the infomation from 2d detection dataset in the 3d segmentation model
You could run download.py to download images first, and then run main_3d_detection.py to do the 3d detection process.
##TO DO 
There is no segmentation part in the model. So if you want to do the segmentation, you may first train the 3d FPN (feature pyramid net) on a 2.5d detection dataset, then add a segmentation branch after the 3d FPN, and fine-tune the FPN.
