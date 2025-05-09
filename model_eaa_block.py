import torch
import torch.nn as nn
import torch.nn.functional as F

# SA Block, not sure if this matches exactly.
class ShuffleAttention(nn.Module):
    def __init__(self, channels, groups=8):
        super(ShuffleAttention, self).__init__()
        assert channels % groups == 0, "Channels must be divisible by groups"
        self.groups = groups
        self.channels_per_group = channels // groups

        # Channel Attention components
        self.channel_fc1 = nn.Linear(self.channels_per_group, self.channels_per_group // 4)
        self.channel_relu = nn.ReLU(inplace=True)
        self.channel_fc2 = nn.Linear(self.channels_per_group // 4, self.channels_per_group)
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention components
        self.spatial_conv = nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size=3, padding=1, groups=self.channels_per_group)
        self.spatial_bn = nn.BatchNorm2d(self.channels_per_group)
        self.spatial_sigmoid = nn.Sigmoid()

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, "Channels must be divisible by groups"

        # Split channels into groups
        x = x.view(batchsize, self.groups, self.channels_per_group, height, width)

        # Channel Attention
        x_channel = x.mean(dim=[3, 4])  # Global Average Pooling: [B, G, C']
        x_channel = self.channel_fc1(x_channel)
        x_channel = self.channel_relu(x_channel)
        x_channel = self.channel_fc2(x_channel)
        x_channel = self.channel_sigmoid(x_channel).unsqueeze(-1).unsqueeze(-1)  # [B, G, C', 1, 1]

        # Spatial Attention
        x_spatial = x.view(batchsize * self.groups, self.channels_per_group, height, width)
        x_spatial = self.spatial_conv(x_spatial)
        x_spatial = self.spatial_bn(x_spatial)
        x_spatial = self.spatial_sigmoid(x_spatial)
        x_spatial = x_spatial.view(batchsize, self.groups, self.channels_per_group, height, width)

        # Combine Channel and Spatial Attention
        out = x * x_channel * x_spatial  # Element-wise multiplication

        # Merge groups and shuffle channels
        out = out.view(batchsize, -1, height, width)
        out = self.channel_shuffle(out)

        return out

# Sketchy implementation of BAA block.  Lots of GPT generated code.
class BAABlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(BAABlock, self).__init__()
        reduced_channels = channels // reduction

        # First block
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected block 1
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        
        # Extra ReLu block.  Added by GPT, allegedly commonly placed between FC layers.
        self.relu = nn.ReLU(inplace=True)
        
        # Fully connected block 2
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
        # Extra BatchNorm block. Added by GPT as a training stabilizer? Probably should be removed.
        self.bn = nn.BatchNorm1d(channels)

        # Final block
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global average pooling
        y = self.global_avg_pool(x).view(b, c)
        # Fully connected layers with ReLU and BatchNorm
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.bn(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # Scale the input features
        return x * y.expand_as(x)
    
class EAA_Module(nn.Module):
    def __init__(self, channels):
        super(EAA_Module, self).__init__()
        self.shuffle_attn_x = ShuffleAttention(channels)
        self.shuffle_attn_y = ShuffleAttention(channels)
        self.baa = BAABlock(channels)

    def forward(self, x, y):
        # x = decoder feature
        # y = encoder feature

        x_sa = self.shuffle_attn_x(x)
        y_sa = self.shuffle_attn_y(y)

        x_baa = self.baa(x)
        gated = y_sa * x_baa  # Element-wise product

        out = x_sa + gated
        return out