import numpy as np
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import einops
import os
from torchvision.transforms.functional import resize


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return ax.imshow(mask_image)


# Inverse the torchvision normalization for visualisation
def inverse_norm(img):
    img = v2.functional.normalize(img, mean=[0.,0.,0.], std=[1/0.229,1/0.224,1/0.225 ])
    img = v2.functional.normalize(img, mean=[-0.485,-0.456,-0.406], std=[1.,1.,1.])
    return img


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))    


def plot_outputs(i, batch, output, args, re_s=False):
    plt.figure(figsize=(10, 5.625))
    if re_s:
        resized_img = resize(batch['img'].to(args.device), [output[0].shape[-2],output[0].shape[-1]], antialias=True)
        rgb_whc = einops.rearrange(inverse_norm(resized_img[0]).numpy(force=True), 'c w h -> w h c')
    else:
        rgb_whc = einops.rearrange(inverse_norm(batch['img'][0]).numpy(force=True), 'c w h -> w h c')
    #mask_whc = einops.rearrange(, 'c w h -> w h c')
    plt.imshow(rgb_whc)
    show_mask(np.round(output[0,0,:,:].numpy(force=True)), plt.gca())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, str(i)) + '.png')
    plt.close()