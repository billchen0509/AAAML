# simple_unet.py

import torch
import torch.nn as nn
from torch.nn import functional as F

def double_conv(in_channels, out_channels):
    '''
    This function creates a module that performs two convolutions followed by a ReLU activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        nn.ReLU(inplace = True)
    )

def crop_tensor(tensor, target_tensor):
    '''
    This function crops the `tensor` to the size of the `target_tensor`
    '''
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = tensor_size - target_size
    delta = delta // 2

    return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_train_1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_train_2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_train_3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_train_4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = 1)

    
    def forward(self, image):
        # encoder
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # decoder
        x = self.up_train_1(x9)
        y = crop_tensor(x7, x)
        x = self.up_conv_1(torch.cat([x, y], 1))
        y = self.up_train_2(x)
        y = crop_tensor(x5, y)
        y = self.up_conv_2(torch.cat([y, y], 1))
        y = self.up_train_3(y)
        y = crop_tensor(x3, y)
        x = self.up_conv_3(torch.cat([y, y], 1))
        x = self.up_conv_4(x)
        y = crop_tensor(x1, x)
        x= self.up_conv_4(torch.cat([x, y], 1))

        # output layer
        out = self.out(x)
        return out

if __name__ == "__main__":
    image = torch.randn((1, 1, 572, 572))
    model = UNet()
    print(model(image).shape)