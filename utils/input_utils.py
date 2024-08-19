import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

'''
Different format annotations returned in XYXY format
'''
def read_txt_box(box_path, image):
    try:
        box_pd = pd.read_csv(box_path, sep=' ', header=None)
    except:
        return None
    input_box = np.array([int(box_pd[1][0]*image.shape[0])-int((box_pd[3][0]*image.shape[0])/2), 
                    int(box_pd[2][0]*image.shape[1])-int((box_pd[4][0]*image.shape[1])/2), 
                    int(box_pd[1][0]*image.shape[0])+int((box_pd[3][0]*image.shape[0])/2),
                    int(box_pd[2][0]*image.shape[1])+int((box_pd[4][0]*image.shape[1])/2), 
                    ])
    return input_box


def read_xml_box(box_path, image):
    box_tree = ET.parse(box_path)
    root = box_tree.getroot()
    #print(root[6][4][0].text)
    #print(image.shape)
    xmin = float(root[6][4][0].text)/640 * image.shape[0]
    ymin = float(root[6][4][1].text)/480 * image.shape[1]
    xmax = float(root[6][4][2].text)/640 * image.shape[0]
    ymax = float(root[6][4][3].text)/480 * image.shape[1]
    
    input_box = np.array([xmin, ymin, xmax, ymax], dtype=int)
    return input_box