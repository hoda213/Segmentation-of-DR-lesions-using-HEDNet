To train the model, run ```python train.py --preprocess '2' --lesion 'NV'``` for training a HEDNet model to segment Neovasularization images of FGADR images. Further technical settings, optimum preprocessing and the highest scores in mAP and PR-AUC will be released and added soon.

The meaning of each preprocessing index is indicated in the following table.

 Preprocessing Index Preprocessing Methods 

 '0'  None 
 '1'  Brightness Balance 
 '2'  Contrast Enhancement 
 '3'  Contrast Enhancement + Brightness Balance 
 '4'  Denoising 
 '5'  Denoising + Brightness Balance 
 '6'  Denoising + Contrast Enhancement 
 '7'  Denoising + Contrast Enhancement + Brightness Balance 
