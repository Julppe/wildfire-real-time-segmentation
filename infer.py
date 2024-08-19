from PIL import Image
import matplotlib.cm as cm
import cv2
import torch
from natsort import natsorted

from utils.plot_outputs import *
from utils.get_args import get_args

from pidnet_utils.configs import config
from models.pidnet import PIDNet, get_seg_model
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from pidnet_utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from pidnet_utils.function import train, validate
from pidnet_utils.utils import create_logger, FullModel


def preprocess(img):
    img = v2.functional.to_image(img)
    img = v2.functional.to_dtype(img, torch.uint8)
    img = v2.functional.resize(img, [1080, 1920], antialias=True)
    img = v2.functional.to_dtype(img, torch.float32, scale=True)
    img = v2.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img


def main():
    # PIDNet outputs
    args = get_args()
    cfg = config
    cfg.MODEL.NAME = 'pidnet_'+args.model_size
    cfg.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_'+args.model_size.capitalize()+'_ImageNet.pth.tar'
    model = get_seg_model(cfg=cfg, imgnet_pretrained=True)
    # Positive weighting for the segmentation loss
    pos_weight = einops.rearrange(torch.tensor([2.5]), '(a b c) -> a b c', a=1, b=1)
    sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bd_criterion = BondaryLoss()

    
    model = FullModel(model, sem_criterion, bd_criterion)
    print('Loading model weights')
    weights = torch.load(os.path.join(args.weight_dir, args.exp+'.pt'), map_location='cpu')
    #print(weights)
    model.load_state_dict(weights, strict=False)
    model.eval()
    model.to('cpu')

    image = Image.open(args.input_image)
    image = preprocess(image)
    # Add batch dim
    image = einops.rearrange(image, ' (a b) c d -> a b c d', a=1)
    #print(image.shape)

    outputs = model(inputs=image, plot_outputs=True)
    
    # Mask 2
    seg2 = torch.round(F.sigmoid(outputs[1][0,0,:,:])).numpy(force=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm.viridis(seg2))
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, args.input_image.split(os.path.sep)[-1]), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()