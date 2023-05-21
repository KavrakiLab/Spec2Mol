import os
import sys
import argparse

import torch
import torch.nn as nn
from torch import optim
import numpy as np

from model1D2Conv import Net1D
from dataset import MSDataset_4channels
from utils import *
import queue

torch.manual_seed(57)



def train(args, train_loader, valid_loader, model, epochs):
    path = os.getcwd()
    epoch = 1
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    criterion = nn.MSELoss()
    model.train()
    epoch_losses = []
    valid_queue = queue.Queue(5)
    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            model.zero_grad()
            spectra = data['spectra'].float().cuda()
            embedding = data['embedding'].cuda()
            preds = model(spectra)
            loss = criterion(preds, embedding)
            epoch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current_lr = scheduler.get_lr()
        print('Learning rate: ', current_lr)
        scheduler.step()
        print('Epoch: ', epoch, ', Train Loss: ', np.mean(epoch_losses))
        epoch_losses = []
        valid_loss = evaluate(valid_loader, model)
        print('Epoch: ', epoch, ', Valid Loss: ', valid_loss)
        if epoch>49 and epoch%5==0 and args.save_models:
            model_dir = path + '/models/model_' + str(epoch) + '.pt'
            torch.save(model.state_dict(),model_dir)
        if epoch>4:
            rem = valid_queue.get()
        valid_queue.put(valid_loss)
        if epoch>4:
            early_stopping(list(valid_queue.queue))
    return model

def early_stopping(losses):
    print(np.std(losses))
    if np.std(losses)<0.0000001:
        sys.exit('Valid loss converging!')
    counter = 0
    if losses[4]>losses[0]:
        counter = counter + 1
    if losses[3]>losses[0]:
        counter = counter + 1
    if losses[2]>losses[0]:
        counter = counter + 1
    if losses[1]>losses[0]:
        counter = counter + 1
    if counter > 4:
        sys.exit('Loss increasing!')
    return


def evaluate(loader, model):
    errors = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            spectra = data['spectra'].float().cuda()
            embedding = data['embedding'].cuda()
            preds = model(spectra)
            error = torch.mean((preds-embedding)**2)
            errors.append(error.item())
    return np.mean(errors)
    


def main(args):

    train_set = MSDataset_4channels('train', args.resolution, args.neg, args.augm, False)
    valid_set = MSDataset_4channels('valid', args.resolution, args.neg)
    test_set = MSDataset_4channels('test', args.resolution, args.neg) 
    print('Number of train data: ', len(train_set))
    print('Number of valid data: ', len(valid_set))
    print('Number of test data: ', len(test_set))

    train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=args.batch_size, 
            num_workers=1)
    valid_loader = torch.utils.data.DataLoader(valid_set, 
            batch_size=64, 
            num_workers=1,
            drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, 
            batch_size=64, 
            num_workers=1)
    data_dim = train_set.get_data_dim()[1]

    print('Data dimensionsality: ', data_dim)

    model = Net1D(args,data_dim).cuda()

    
    print(model)
    print('Number of model parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = train(args, train_loader, valid_loader, model, args.num_epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_epochs', type=int, default=201, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.004, help='Learning rate')
    parser.add_argument('-reg', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('-batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('-valid_size', type=int, default=512, help='Number of molecules in validset')
    parser.add_argument('-batch_size_valid', type=int, default=128, help='Batch size for valid')
    parser.add_argument('-channels_med_1', type=int, default=4, help='Number of channels after first conv layer')
    parser.add_argument('-channels_out', type=int, default=8, help='Number of output channels in the last conv layer')
    parser.add_argument('-conv_kernel_dim_1', type=int, default=200, help='Kernel size for first conv layer')
    parser.add_argument('-conv_kernel_dim_2', type=int, default=200, help='Kernel size for second conv layer')
    parser.add_argument('-conv_stride_1', type=int, default=1, help='Stride for first conv layer')
    parser.add_argument('-conv_stride_2', type=int, default=1, help='Stride for second conv layer')
    parser.add_argument('-conv_dilation', type=int, default=1, help='Dilation for conv layers')
    parser.add_argument('-conv_padding_1', type=int, default=100, help='Padding for first conv layer')
    parser.add_argument('-conv_padding_2', type=int, default=100, help='Padding for second conv layer')
    parser.add_argument('-pool_kernel_dim_1', type=int, default=50, help='Kernel size for first pool layer')
    parser.add_argument('-pool_kernel_dim_2', type=int, default=50, help='Kernel size for second pool layer')
    parser.add_argument('-pool_stride_1', type=int, help='Stride for first pool layer')
    parser.add_argument('-pool_stride_2', type=int, help='Stride for second pool layer')
    parser.add_argument('-pool_dilation', type=int, default=1, help='Dilation for pool layers')
    parser.add_argument('-pool_padding_1', type=int, default=0, help='Padding for first pool layers')
    parser.add_argument('-pool_padding_2', type=int, default=0, help='Padding for second pool layers')
    parser.add_argument('-fc_dim_1', type=int, default=512, help='Size of the fully connected layer')
    parser.add_argument('-conv_layers', type=int, default=2, help='Number of conv layers')
    parser.add_argument('-resolution', type=int, default=2, help='Number of decimal points for mass')
    parser.add_argument('-save_embs', type=bool, default=False, help='Save embeddings for train/valid/test data')
    parser.add_argument('-save_models', type=bool, default=True, help='Save trained models')
    parser.add_argument('-neg', type=bool, default=True, help='Use negative mode spectra if true')
    parser.add_argument('-emb_dim', type=int, default=512, help='Dimensionality of the embedding space')
    parser.add_argument('-augm', type=bool, default=True, help='Perorm data augmentation if true')

    args = parser.parse_args()

    args.pool_stride_1 = args.pool_kernel_dim_1
    args.pool_stride_2 = args.pool_kernel_dim_2

    if not args.neg:  # if negative mode spectra is not used then the channels are two (high and low energy positive spectra)
        args.channels_in = 2
    else:
        args.channels_in = 4

    main(args)