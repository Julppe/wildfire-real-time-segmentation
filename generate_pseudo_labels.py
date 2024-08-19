import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from natsort import natsorted
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="checkpoints/sam_vit_h_4b8939.pth", help='Model checkpoint path.')
    parser.add_argument('--model', default="vit_h", help='SAM model variant.')
    parser.add_argument('--device', default="cuda", help='Device, cuda or cpu')
    parser.add_argument('--dir', default='data', help='Data location directory')
    parser.add_argument('--label-format', default='txt')
    parser.add_argument('--n-vis', default=8, type=int, help='Number of samples to visualise in the end of mask generation.')
    parser.add_argument('--mode', default='test', help='Datasplit part to generate. Options: train, valid, test. Default=valid.')
    parser.add_argument('--model-type', default="vit_h", help='SAM model type to use for mask generation. Default=vit_h')
    parser.add_argument('--only-vis', action='store_true', help='Only visualise masks.')
    parser.add_argument('--img-format', default='.jpg')
    parser.add_argument('--output-dir', default='pseudo_labels')
    args = parser.parse_args()
    return args


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_model(args):
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=args.device)

    predictor = SamPredictor(sam)
    return predictor


def read_txt_box(box_path, image):
    try:
        box_pd = pd.read_csv(box_path, sep=' ', header=None)
    except:
        return None    
    input_box = np.array([int(box_pd[1][0]*image.shape[1])-int((box_pd[3][0]*image.shape[1])/2), 
                    int(box_pd[2][0]*image.shape[0])-int((box_pd[4][0]*image.shape[0])/2), 
                    int(box_pd[1][0]*image.shape[1])+int((box_pd[3][0]*image.shape[1])/2),
                    int(box_pd[2][0]*image.shape[0])+int((box_pd[4][0]*image.shape[0])/2), 
                    ])
    return input_box


def read_xml_box(box_path, image):
    box_tree = ET.parse(box_path)
    root = box_tree.getroot()
    #print(root[6][4][0].text)
    xmin = float(root[6][4][0].text)
    ymin = float(root[6][4][1].text)
    xmax = float(root[6][4][2].text)
    ymax = float(root[6][4][3].text)
    
    input_box = np.array([xmin, ymin, xmax, ymax], dtype=int)
    return input_box


def read_img(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
    

def read_mask(mask_path):
    image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.array(image/255, dtype=int)


def generate_masks(args, predictor, mode='train'):
    img_dir = os.path.join(args.dir, 'images', mode)
    box_dir = os.path.join(args.dir, 'labels', mode)
    masks_dir = os.path.join(args.output_dir)
    label_list = os.listdir(box_dir)
    txt_list = [x[:-4] for x in label_list if x[-4:] == '.txt']
    xml_list = [x[:-4] for x in label_list if x[-4:] == '.xml']

    for img_path in natsorted(os.listdir(img_dir)):
        image = read_img(os.path.join(img_dir, img_path))
        
        predictor.set_image(image)
        if img_path.split('.')[0] in txt_list:
            input_box = read_txt_box(os.path.join(box_dir, img_path.split('.')[0]+'.txt'), image)
        elif img_path.split('.')[0] in xml_list:
            input_box = read_xml_box(os.path.join(box_dir, img_path.split('.')[0]+'.xml'), image)
        if not isinstance(input_box, np.ndarray):
            print("No bounding box found, leaving mask ungenerated.")
            print("Image: "+img_path)
        else:
            masks, _, _ = predictor.predict(
                                point_coords=None,
                                point_labels=None,
                                box=input_box[None, :],
                                multimask_output=False,
                                )
            
            mask_img = np.array(masks[0], dtype=int)*255
            cv2.imwrite(os.path.join(masks_dir, img_path.split('.')[0]+'.png'), mask_img)



def visualise_random(args, mode='test', n=16):
    img_dir = os.path.join(args.dir, 'images', mode)
    box_dir = os.path.join(args.dir, 'labels', mode)
    masks_dir = os.path.join(args.output_dir)
    fig, ax = plt.subplots(16//4, 4)

    for i, mask_path in enumerate(np.random.choice(natsorted(os.listdir(masks_dir)), 16)):
        img_path = os.path.join(img_dir, mask_path[:-4]+args.img_format)
        image = read_img(img_path)
        current_ax = ax[i//4, i%4]
        current_ax.imshow(image)
        mask = read_mask(os.path.join(masks_dir, mask_path))
        show_mask(mask, current_ax)
        if args.label_format == 'txt':
            box = read_txt_box(os.path.join(box_dir, mask_path[:-4]+'.txt'), image)
        elif args.label_format == 'xml':
            box = read_xml_box(os.path.join(box_dir, mask_path[:-4]+'.xml'), image)
        show_box(box, current_ax)
        current_ax.axis('off')
    plt.show()


def main():
    args = parse_args()
    sam_checkpoint = args.checkpoint
    model_type = args.model
    device = args.device
    np.random.seed(42)
    predictor = get_model(args)
    if not args.only_vis:
        generate_masks(args, predictor, mode=args.mode)
    visualise_random(args, mode=args.mode)


if __name__ == '__main__':
    main()