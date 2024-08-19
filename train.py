import torch
from datasets.segmentation_data import WFSeg
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from utils.get_args import get_args
from eval import eval
from models.pidnet import PIDNet, get_seg_model
from pidnet_utils.configs import config
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from pidnet_utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from pidnet_utils.function import train, validate
from pidnet_utils.utils import create_logger, FullModel
import einops
import matplotlib.pyplot as plt


def write_logs(epoch, val_loss, model, optim, args, miou=0):
    # Create a new log file if there isn't one
    try:
        if epoch == 0:
            df = pd.DataFrame(columns=['epoch', 'val_loss', 'mIoU'])
        else:
            df = pd.read_csv(os.path.join(args.log_dir, args.exp+'.csv'))
    except:
        df = pd.DataFrame(columns=['epoch', 'val_loss', 'mIoU'])

    # Save weights if the best loss is achieved
    if epoch == 0 or val_loss < np.amin(df['val_loss']):
        print('Best validation loss so far, saving weights.')
        save_weights(model, optim, args)
    if epoch == 0 or miou > np.amax(df['mIoU']):
        print('Best mIoU so far, saving weights.')
        save_weights(model, optim, args, miou=True)

    # Add new loss to df
    df.loc[len(df)] = [epoch, val_loss, miou]
    # Save logs
    df.to_csv(os.path.join(args.log_dir, args.exp+'.csv'), index=False)


def save_weights(model, optim, args, miou=False):
    if miou:
        torch.save(model.state_dict(), os.path.join(args.weight_dir, args.exp+'_miou.pt'))
        torch.save(optim.state_dict(), os.path.join(args.weight_dir, args.exp+'_optim_miou.pt'))
    else:
        torch.save(model.state_dict(), os.path.join(args.weight_dir, args.exp+'.pt'))
        torch.save(optim.state_dict(), os.path.join(args.weight_dir, args.exp+'_optim.pt'))


def train(model, trainloader, valloader, args):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print('Starting training')
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        loss = 0
        optim.zero_grad()
        for i, batch in enumerate(trainloader):
            if args.test_val:
                break
            input = batch['img'].to(args.device)
            if not args.boxsup:
                mask = batch['mask'].to(args.device)
                boundary = batch['boundary'].to(args.device)
                losses, _, acc, loss_list = model(input, mask, boundary)
            else:
                loss = model(input, labels=None, bd_gt=None, id=None, 
                             box=batch['box'].to(args.device), 
                             lab_img=batch['lab_img'].to(args.device))
            #print(losses)
            loss = losses.mean()
            #acc = acc.mean()
            
            last_loss = losses.item()
            epoch_loss += last_loss
            loss.backward()
            optim.step()
            optim.zero_grad()
        
            if i % 10 == 0 and args.verbose:
                print('Epoch: {:d}, batch: {:d}, Last training loss: {:.4f}'.format(epoch, i, last_loss))
        
        print('Finished epoch: {:d}, training loss: {:.7f}. validating'.format(epoch, (epoch_loss/(i+1))))
        epoch_loss = 0
        # Validate
        loss_dict = eval(model, args, valloader)
        val_loss = loss_dict['mean_val_loss']
        miou = loss_dict['mean_iou']
        print('Finished validating, validation loss: {:.7f}'.format(val_loss))
        print('Mean index over union: {:.7f}'.format(miou))
        # Write logs and save weights
        write_logs(epoch, val_loss, model, optim, args, miou)


def main():
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, args.exp)
    print(args)
    np.random.seed(42)
    config.MODEL.NAME = 'pidnet_'+args.model_size
    config.MODEL.PRETRAINED = 'pretrained_models/imagenet/PIDNet_'+args.model_size.capitalize()+'_ImageNet.pth.tar'
    model = get_seg_model(cfg=config, imgnet_pretrained=True)
    # Positive weighting for the segmentation loss
    pos_weight = einops.rearrange(torch.tensor([2.5]), '(a b c) -> a b c', a=1, b=1)
    sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bd_criterion = BondaryLoss()
    
    model = FullModel(model, sem_criterion, bd_criterion)
    model.to(args.device)
    
    trainset = WFSeg(root_dir=args.data_dir, 
                     mode='train',
                     boundary=(not args.boxsup),
                     include_id=args.include_id,
                     args=args)
    valset = WFSeg(root_dir=args.data_dir, 
                    mode='valid',
                    boundary=(not args.boxsup),
                     include_id=args.include_id,
                     args=args)
    
    # print(len(trainset))
    # print(len(valset))
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    if args.debug_val:
        loss_dict = eval(model, args, valloader)
        print(loss_dict)
        exit()
    train(model, trainloader=trainloader, valloader=valloader, args=args)


if __name__ == "__main__":
    main()