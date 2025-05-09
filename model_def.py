import torch
import torch.nn as nn
import torch.nn.functional as F
from model_eaa_block import EAA_Module

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# --- EAAU-Net Model ---
class EAAUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(EAAUNet, self).__init__()

        # Conv-1 (480x480 → 480x480)
        self.conv1 = ConvBlock(in_channels, features)  # e1

        # Stage-1 (no downsampling; 480x480 → 480x480)
        self.stage1 = ConvBlock(features, features)    # e1 again if reused

        # Stage-2 (downsample: 480x480 → 240x240)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = ConvBlock(features, features * 2)  # e2

        # Bottleneck (downsample: 240x240 → 120x120)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features * 2, features * 4)  # b

        # Decoder: Upstage-2 (120x120 → 240x240)
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.eaa2 = EAA_Module(features * 2)
        self.upstage2 = ConvBlock(features * 4, features * 2)

        # Decoder: Upstage-1 (240x240 → 480x480)
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.eaa1 = EAA_Module(features)
        self.upstage1 = ConvBlock(features * 2, features)

        # Output
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)                    # Conv-1
        e1 = self.stage1(e1)                  # Stage-1 (no downsample)
        e2 = self.stage2(self.pool1(e1))      # Stage-2
        b = self.bottleneck(self.pool2(e2))   # Bottleneck

        # Decoder
        d2 = self.up2(b)                      # Upsample
        eaa2 = self.eaa2(e2, d2)              # EAA (1)
        d2 = self.upstage2(torch.cat([eaa2, d2], dim=1))  # Upstage-2

        d1 = self.up1(d2)                     # Upsample
        eaa1 = self.eaa1(e1, d1)              # EAA (2)
        d1 = self.upstage1(torch.cat([eaa1, d1], dim=1))  # Upstage-1

        return self.output_conv(d1)           # Output

if __name__ == "__main__":
    model = EAAUNet()
    print(model)