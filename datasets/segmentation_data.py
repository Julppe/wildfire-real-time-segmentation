import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
import einops
from natsort import natsorted
import cv2
from itertools import compress
from utils.input_utils import *
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from skimage import color
#import detectron2


class WFSeg(Dataset):
    '''Wildfire smoke segmentation dataset'''
    def __init__(self, 
                 root_dir,
                 mode='train',
                 boundary=False,
                 include_id=False,
                 manual_masks=False,
                 args=None):
        self.mode = mode
        self.img_dir = os.path.join(root_dir, 'images', self.mode)
        self.img_list = natsorted(os.listdir(self.img_dir))
        # If using manual masks, only include samples with a mask in the dataset
        if manual_masks:
            self.mask_dir = os.path.join(root_dir, args.eval_dir, self.mode)
            self.mask_list = [x[:-4] for x in natsorted(os.listdir(self.mask_dir))]
            in_imgs = lambda x, a: (x[:-4] in a) or (x[:-5] in a)
            in_masks = [in_imgs(x, self.mask_list) for x in self.img_list]
            self.img_list = list(compress(self.img_list,in_masks))
        # If evaluating the sam masks, use the sam masks but filter only
        # for images which also have a manual mask
        elif args.eval_sam or args.eval_snake:
            # Filter img list based on manual masks
            self.mask_dir = os.path.join(root_dir, 'manual_masks', self.mode)
            self.mask_list = [x[:-4] for x in natsorted(os.listdir(self.mask_dir))]
            in_imgs = lambda x, a: (x[:-4] in a) or (x[:-5] in a)
            in_masks = [in_imgs(x, self.mask_list) for x in self.img_list]
            self.img_list = list(compress(self.img_list,in_masks))
            # Filter mask list based on manual masks
            self.manual_list = self.mask_list.copy()
            if args.eval_snake:
                self.mask_dir = os.path.join(root_dir, args.sup_dir, self.mode)
            else:
                self.mask_dir = os.path.join(root_dir, 'sam_masks', self.mode)
            self.mask_list = [x[:-4] for x in natsorted(os.listdir(self.mask_dir))]
            mask_in_masks = [(x in self.manual_list) for x in self.mask_list]
            #print(len(self.mask_list))
            self.mask_list = list(compress(self.mask_list, mask_in_masks))
        else:
            self.mask_dir = os.path.join(root_dir, args.sup_dir, self.mode)
            self.mask_list = [x[:-4] for x in natsorted(os.listdir(self.mask_dir))]
            # in_imgs = lambda x, a: (x[:-4] in a) or (x[:-5] in a)
            # in_masks = [in_imgs(x, self.mask_list) for x in self.img_list]
            # self.img_list = list(compress(self.img_list,in_masks))
            # self.manual_list = self.mask_list.copy()
            # mask_in_masks = [(x in self.manual_list) for x in self.mask_list]
            # self.mask_list = list(compress(self.mask_list, mask_in_masks))

        self.geometric_augments, self.color_augments = self.init_augments()
        self.boundary = boundary
        self.include_id = args.include_id
        self.include_box = args.include_box or args.boxsup
        self.boxsup = args.boxsup

        if self.include_box:
            self.box_format = args.label_format
            self.box_dir = os.path.join(root_dir, 'labels', self.mode)
            self.box_list = natsorted(os.listdir(self.box_dir))
            #self.box_list = list(compress(self.box_list, in_masks))


    def __len__(self):
        return len(self.img_list) #- 1
    
    def init_augments(self):
        geometric_augments = {}
        geometric_augments['random_crop'] = v2.RandomCrop(size=[955, 1680], padding=None) # 80% Crop to keep it similar with the inference data
        geometric_augments['vertical_flip'] = v2.RandomVerticalFlip(p=1)
        geometric_augments['rotate'] = v2.RandomRotation(30)
        geometric_augments['perspective'] = v2.RandomPerspective(0.2,p=1)
        geometric_augments['erasing'] = v2.RandomErasing(p=1)

        color_augments = {}
        color_augments['grayscale'] = v2.Grayscale(num_output_channels=3)
        color_augments['gaussian_blur'] = v2.GaussianBlur(7)
        color_augments['invert'] = v2.RandomInvert(p=1)
        color_augments['sharpness_lower'] = v2.RandomAdjustSharpness(0.5,p=1)
        color_augments['sharpness_higher'] = v2.RandomAdjustSharpness(1.5,p=1)
        color_augments['color_jitter'] = v2.ColorJitter(0.2,0.2,0.2,0.2)

        return geometric_augments, color_augments
    
    def apply_random_augment(self, mask, img, augment_list=['none',
                                                      'crop',
                                                      'vertical_flip',
                                                      'rotate',
                                                      'perspective',
                                                      'erasing',
                                                      'grayscale',
                                                      'gaussian_blur',
                                                      'invert',
                                                      'sharpness_lower',
                                                      'sharpness_higher',
                                                      'color_jitter',
                                                      ], 
                                                      p=[0.1]+np.repeat([0.9/15],11).tolist(), # Slightly higher probability for no augmentation
                                                      empty_mask=False,
                                                      edge=None,
                                                      box=None):
        augment_name = np.random.choice(augment_list, 1, p)[0]
        #print(augment_name)

        if augment_name in self.geometric_augments and not empty_mask:
            augment = self.geometric_augments[augment_name]
            stacked_output = augment(torch.cat([mask, img], dim=0))
            mask = stacked_output[:1]
            img = stacked_output[1:4]
        
        elif augment_name in self.geometric_augments:
            augment = self.geometric_augments[augment_name]
            img = augment(img)
        
        elif augment_name in self.color_augments:
            augment = self.color_augments[augment_name]
            img = augment(img)

        if augment_name in self.geometric_augments and self.boundary:
            edge = augment(edge)
            #return img, mask, augment_name, edge
        
        if augment_name in self.geometric_augments and self.include_box:
            #print(box)
            box = augment(box)

        return img, mask, augment_name, edge, box

    
    def transform(self, sample, empty_mask=False):
        img = sample['img']
        mask = sample['mask']
        if self.boxsup:
            box = sample['box']
        else:
            box = None
        if self.boundary:
            edge = sample['boundary']
        
        # For training apply horizontal flip 50% of the time
        horizontal_flip = (np.random.random() > 0.5)
        
        # For training also apply a random augmentation
        if self.mode == 'train':
            # Image resize and norm
            img = v2.functional.to_image(img)
            img = v2.functional.to_dtype(img, torch.uint8)
            # Mask norm
            mask = v2.functional.to_image(mask)
            mask = v2.functional.to_dtype(mask, torch.uint8)

            if self.boundary:
                edge = einops.rearrange(edge, 'h (w c) -> h w c ', c=1)
                edge = v2.functional.to_image(edge)
                edge = v2.functional.to_dtype(edge, torch.uint8)
            
            else:
                edge = None

            if horizontal_flip:
                img = v2.functional.horizontal_flip(img)
                if not empty_mask:
                    mask = v2.functional.horizontal_flip(mask)
                if self.boundary:
                    edge = v2.functional.horizontal_flip(edge)

            # Apply a random augmentation
            img, mask, augment_name, edge, box = self.apply_random_augment(mask, img, augment_list=['none',
                                                      'crop',
                                                      'vertical_flip',
                                                      'rotate',
                                                      'perspective',
                                                      'erasing',
                                                      'grayscale',
                                                      'gaussian_blur',
                                                      'invert',
                                                      'sharpness_lower',
                                                      'sharpness_higher',
                                                      'color_jitter',
                                                      ], 
                                                      p=[0.1]+np.repeat([0.9/15],11).tolist(),
                                                      empty_mask=empty_mask,
                                                      edge=edge,
                                                      box=box)

            img = v2.functional.resize(img, [1080, 1920], antialias=True)
            img = v2.functional.to_dtype(img, torch.float32, scale=True)
            img = v2.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            mask = v2.functional.resize(mask, [1080, 1920], antialias=True)
            mask = v2.functional.to_dtype(mask, torch.float32, scale=True)
            mask = torch.round(mask)

            if self.boundary:
                edge = v2.functional.resize(edge, [1080, 1920], antialias=True)
                edge = v2.functional.to_dtype(edge, torch.float32, scale=True)
                return {'img':img, 'mask':mask, 'boundary':edge}
            
            elif self.boxsup:
                return {'img':img, 'box':box}
            
        # For validation and testing only resize and normalize
        else:
            # Image resize and norm
            img = v2.functional.to_image(img)
            img = v2.functional.to_dtype(img, torch.uint8)
            img = v2.functional.resize(img, [1080, 1920], antialias=True)
            img = v2.functional.to_dtype(img, torch.float32, scale=True)
            img = v2.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            # Mask norm
            mask = v2.functional.to_image(mask)
            mask = v2.functional.to_dtype(mask, torch.uint8)
            mask = v2.functional.resize(mask, [1080, 1920], antialias=True)
            mask = v2.functional.to_dtype(mask, torch.float32, scale=True)
            mask = torch.round(mask)

            # Boundary
            if self.boundary:
                edge = einops.rearrange(edge, 'h (w c) -> h w c ', c=1)
                edge = v2.functional.to_image(edge)
                edge = v2.functional.to_dtype(edge, torch.uint8)
                edge = v2.functional.resize(edge, [1080, 1920], antialias=True)
                edge = v2.functional.to_dtype(edge, torch.float32, scale=True)
                return {'img':img, 'mask':mask, 'boundary':edge}

        return {'img':img, 'mask':mask}
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.img_dir, self.img_list[idx]))
        image = image.convert("RGB")
        #print(self.mask_list)

        if self.img_list[idx][-4:] == '.jpg':
            mask_name = self.img_list[idx][:-4]
        else:
            mask_name = self.img_list[idx][:-5]
        
        if mask_name not in self.mask_list:
            mask = Image.fromarray(np.zeros((1080, 1920)))
            empty_mask = True
        else:
            mask = Image.open(os.path.join(self.mask_dir, mask_name+'.png'))
            mask = mask.convert('L')
            empty_mask = False

        if self.boundary:
            #print(np.array(mask).shape)
            edge = cv2.Canny(np.uint8(np.array(mask)), 0.1, 0.2)
            kernel = np.ones((4, 4), np.uint8)
            edge = edge[6:-6, 6:-6]
            edge = np.pad(edge, ((6,6),(6,6)), mode='constant')
            edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0

            sample = {'img':image, 'mask':mask, 'boundary':edge}
            #print(image.mode)
            sample = self.transform(sample, empty_mask=empty_mask)

        else:
            sample = {'img':image, 'mask':mask}
            sample = self.transform(sample, empty_mask=empty_mask)

        if self.include_id:
            sample['id'] = self.img_list[idx][:-4]
            sample['empty_mask'] = empty_mask

        if self.include_box and not self.boxsup:
            if self.box_format=='txt':
                box_img = einops.rearrange(sample['img'], 'c h w -> h w c')
                box = read_txt_box(os.path.join(self.box_dir, self.box_list[idx]), box_img)
                
            elif self.box_format=='xml':
                box_img = einops.rearrange(sample['img'], 'c h w -> h w c')
                box = read_xml_box(os.path.join(self.box_dir, self.box_list[idx]), box_img)

            box = tv_tensors.BoundingBoxes(box, 
                                         format=tv_tensors.BoundingBoxFormat.XYXY,
                                         spatial_size=F.get_spatial_size(image))
            sample['box'] = box
            #print(sample['box'])

        return sample