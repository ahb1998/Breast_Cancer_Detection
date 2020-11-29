# Breast_Cancer_Detection
Breast cancer detection by Python. Deep learning using Keras and U-NET algorithm.

## what is the U-NET algorithm ?
This algorithm mostly used for "Medical Image" semantic segmentation.
The UNET was developed by Olaf Ronneberger et al.

### What pictures i used?
I used breast tissue pictures from https://portal.gdc.cancer.gov/  
It is about 58 pictures. I used 56 for training and 2 for testing my model.
All of the pictures had Mask layer(picture) that showed the cancer cells.


## How it works?
1- data entry: pictures path included.  
2- Preprocessing datas includes cropping ,...  
3-showing results in Preprocessing stage  
4-Implementation of U_NET Model for Semantic Segmentation  
5-Define U_NET Model Evaluator (Intersection Over Union _ IOU)  
6-Show The Results per Epoch  
7-Train U_NET Model using Training Samples  
8-U_NET Model Evaluation using Test Samples  
9-Show Final Results (Segmented Images)  
10-Show Loss and IOU Plots  
