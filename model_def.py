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

        # Conv-1
        self.conv1 = ConvBlock(in_channels, features)  # e1

        # Stage-1: It is not specified what exactly this is supposed to be.
        # Assume that this is a downsample, like a normal U-net.
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage1 = ConvBlock(features, features * 2)  # e2

        # Stage-2: Another down-block.
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = ConvBlock(features * 2, features * 4)  # e3

        # Bottleneck
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(features * 4, features * 8)  # b

        # EAA after Bottleneck -> Upstage-2
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.eaa3 = EAA_Module(features * 4)
        self.upstage2 = ConvBlock(features * 8, features * 4)  # Decoder block after EAA3

        # EAA after Upstage-2 -> Upstage-1
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.eaa2 = EAA_Module(features * 2)
        self.upstage1 = ConvBlock(features * 4, features * 2)

        # EAA after Upstage-1 -> Conv-1
        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.predict = ConvBlock(features * 2, features)

        # Output
        self.output_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)                  # Conv-1
        e2 = self.stage1(self.pool1(e1))    # Stage-1
        e3 = self.stage2(self.pool2(e2))    # Stage-2

        # Bottleneck
        b = self.bottleneck(self.pool3(e3)) # Bottleneck

        # Decoder with EAA blocks
        d3 = self.up3(b)
        eaa3 = self.eaa3(e3, d3)            # EAA (1)
        d3 = self.upstage2(torch.cat([eaa3, d3], dim=1))  # Upstage-2

        d2 = self.up2(d3)
        eaa2 = self.eaa2(e2, d2)            # EAA (2)
        d2 = self.upstage1(torch.cat([eaa2, d2], dim=1))  # Upstage-1

        d1 = self.up1(d2)
        d1 = self.predict(torch.cat([eaa2, d1], dim=1))   # Predict

        return self.output_conv(d1)         # Output

if __name__ == "__main__":
    model = EAAUNet()
    print(model)