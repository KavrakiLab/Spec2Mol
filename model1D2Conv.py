import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable



def conv_out_dim(length_in, kernel, stride, padding, dilation):
    length_out = (length_in + 2 * padding - dilation * (kernel - 1) - 1)// stride + 1
    return length_out


class Net1D(nn.Module):
    def __init__(self, args, length_in):
        super(Net1D, self).__init__()
        out_1 = conv_out_dim(length_in, args.conv_kernel_dim_1, args.conv_stride_1, args.conv_padding_1, args.conv_dilation)
        out_2 = conv_out_dim(out_1, args.pool_kernel_dim_1, args.pool_stride_1, args.pool_padding_1, args.pool_dilation)
        out_3 = conv_out_dim(out_2, args.conv_kernel_dim_2, args.conv_stride_2, args.conv_padding_2, args.conv_dilation)
        out_4 = conv_out_dim(out_3, args.pool_kernel_dim_2, args.pool_stride_2, args.pool_padding_2, args.pool_dilation)
        self.cnn_out = args.channels_out*out_4
        self.conv1 = nn.Conv1d(args.channels_in, args.channels_med_1, args.conv_kernel_dim_1, stride=args.conv_stride_1, padding=args.conv_padding_1, dilation=args.conv_dilation)
        self.norm1 = nn.BatchNorm1d(args.channels_med_1)
        self.pool1 = nn.MaxPool1d(args.pool_kernel_dim_1, stride=args.pool_stride_1, padding=args.pool_padding_1, dilation=args.pool_dilation)
        self.pool2 = nn.MaxPool1d(args.pool_kernel_dim_2, stride=args.pool_stride_2, padding=args.pool_padding_2, dilation=args.pool_dilation)
        self.conv2 = nn.Conv1d(args.channels_med_1, args.channels_out, args.conv_kernel_dim_2, stride=args.conv_stride_2, padding=args.conv_padding_2, dilation=args.conv_dilation)
        self.norm2 = nn.BatchNorm1d(args.channels_out)
        self.fc1 = nn.Linear(args.channels_out*out_4, args.fc_dim_1)
        self.fc2 = nn.Linear(args.fc_dim_1, args.emb_dim)
        self.norm3 = nn.BatchNorm1d(args.fc_dim_1)


    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1, self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        return x