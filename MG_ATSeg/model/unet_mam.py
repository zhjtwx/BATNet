import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, inp):
        return self.conv(inp)


class MAMAttention(nn.Module):
    def __init__(self, in_channels):
        super(MAMAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map with shape [batch, channels, height, width]

        Returns:
            torch.Tensor: Attention-modulated feature map
        """
        # Mirror the feature map along the width dimension (left-right axis)
        x_mirrored = torch.flip(x, dims=[-1])

        # Process original and mirrored features independently
        att_original = self.sigmoid(self.conv(x))
        att_mirrored = self.sigmoid(self.conv(x_mirrored))

        # Logical AND operation (implemented as multiplication)
        contralateral_att = att_original * att_mirrored

        # Modulate original features with attention
        modulated_features = x * contralateral_att

        return modulated_features


class MAMUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MAMUnet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.mam3 = MAMAttention(256)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.mam4 = MAMAttention(512)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.mam6 = MAMAttention(512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.mam7 = MAMAttention(256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        m3 = self.mam3(p3)
        c4 = self.conv4(m3)
        p4 = self.pool4(c4)
        m4 = self.mam4(p4)
        c5 = self.conv5(m4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        m6 = self.mam6(c6)
        up_7 = self.up7(m6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        m7 = self.mam7(c7)
        up_8 = self.up8(m7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # print(c10.shape)
        out = nn.Softmax()(c10)
        return out

# x = torch.rand(4, 5, 512,512)
#
# y = MAMUnet(5, 2)
# for i in y.parameters():
#     print(i)
# print(y(x))
