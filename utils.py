import os
import glob
from preprocess import clahe_gridsize
import cv2
import config
import torch.nn as nn
lesion_name = config.LESION_NAME

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

train_ratio = 0.8
eval_ratio = 0.2

def get_images(image_dir, preprocess, phase):
    if phase == 'train' or phase == 'eval' :
        setname = 'TrainingSet'
    elif phase == 'test':
        setname = 'TestingSet' 
    
    limit = 2
    grid_size = 8
    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess))

    if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname)):
        os.mkdir(os.path.join(image_dir, 'Images_CLAHE' + preprocess, setname))

        # compute mean brightess
        meanbright = 0.
        images_number = 0
        for tempsetname in ['TrainingSet', 'TestingSet']:
            imgs_ori = glob.glob(image_dir  + 'OriginalImages/'+ tempsetname + '/*.png')
            imgs_ori.sort()
            images_number += len(imgs_ori)
            # mean brightness.
            for img_path in imgs_ori:
                
                img_name = os.path.split(img_path)[-1].split('.')[0]

                mask_path = image_dir + 'Groundtruths/'+ tempsetname+ '/Mask/'+ img_name + '.png'
                gray = cv2.imread(img_path, 0)
                mask_img = cv2.imread(mask_path, 0)
                brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                meanbright += brightness
        meanbright /= images_number
        
        imgs_ori = glob.glob(image_dir+ 'OriginalImages/' + setname + '/*.png')
        preprocess_dict = {'0': [False, False, None], '1': [False, False, meanbright], '2': [False, True, None], '3': [False, True, meanbright], '4': [True, False, None], '5': [True, False, meanbright], '6': [True, True, None], '7': [True, True, meanbright]}
        for img_path in imgs_ori:
            img_name = os.path.split(img_path)[-1].split('.')[0]
            mask_path = image_dir+ 'Groundtruths/'+ setname+ '/Mask/'+ img_name + '.png'
            clahe_img = clahe_gridsize(img_path, mask_path, denoise=preprocess_dict[preprocess][0], contrastenhancement=preprocess_dict[preprocess][1], brightnessbalance=preprocess_dict[preprocess][2], cliplimit=limit, gridsize=grid_size)
            cv2.imwrite(image_dir+ 'Images_CLAHE' + preprocess+'/'+ setname+'/'+ os.path.split(img_path)[-1], clahe_img)
        
    imgs = glob.glob(image_dir+ 'Images_CLAHE' + preprocess+'/'+ setname+'/'+ '*.png')

    imgs.sort()
    mask_paths = []
    train_number = int(len(imgs) * train_ratio)
    eval_number = int(len(imgs) * eval_ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs

    mask_path = image_dir +'Groundtruths'+'/'+ setname

    lesions = ['Mask']
    lesion_abbvs = [lesion_name]
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):

            candidate_path = mask_path+'/'+ lesion+'/'+ name +  '.png'
            if os.path.exists(candidate_path):
                paths.append(candidate_path)

            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths
