import torch
import torch.nn as nn

######################################
# 3 blocks
######################################

def conv2d_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
#         nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
#         nn.BatchNorm2d(out_channels),
    )


class Encoder(nn.Module):
    def __init__(self, f1=16, depth=6, mode="cascade"):
        super(Encoder, self).__init__()
        
        self.mode = mode
        self.depth = depth
        
        self.dconv1 = conv2d_block(1, f1)
        self.dconv2 = conv2d_block(f1, f1*2)
        self.dconv3 = conv2d_block(f1*2, f1*4)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.dropout = nn.Dropout()

#         self.bottle = conv2d_block(f1*4, f1*8)
        
        nf_bottle = f1*8
        self.bottle1 = nn.Conv2d(f1*4, nf_bottle, kernel_size=3, stride=1, padding=1)
        self.bottle2 = nn.Conv2d(nf_bottle, nf_bottle, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bottle3 = nn.Conv2d(nf_bottle, nf_bottle, kernel_size=3, stride=1, padding=4, dilation=4)
        self.bottle4 = nn.Conv2d(nf_bottle, nf_bottle, kernel_size=3, stride=1, padding=8, dilation=8)
        self.bottle5 = nn.Conv2d(nf_bottle, nf_bottle, kernel_size=3, stride=1, padding=16, dilation=16)
        self.bottle6 = nn.Conv2d(nf_bottle, nf_bottle, kernel_size=3, stride=1, padding=32, dilation=32)
        self.bottle_list = [self.bottle1, self.bottle2, self.bottle3, self.bottle4, self.bottle5, self.bottle6]

    def forward(self, x):

        conv1 = self.dconv1(x)
        x1 = self.maxpool(conv1)

        conv2 = self.dconv2(x1)
        x2 = self.maxpool(conv2)
        
        conv3 = self.dconv3(x2)
        x3 = self.maxpool(conv3)
#         x3 = self.dropout(x3)
        
#         enc_out = self.bottle(x3)

        bottle = self.bottle_list[0](x3)
        
        dilated_layers = []
        dilated_layers.append(bottle)
        if self.mode == "cascade":
            for i in range(self.depth-1):
                bottle = self.bottle_list[i+1](bottle)
                dilated_layers.append(bottle)
        
        enc_out = sum(dilated_layers)
        
        return enc_out, conv3, conv2, conv1


class Decoder(nn.Module):
    def __init__(self, f1=16):
        super(Decoder, self).__init__()
        
        self.up1 = nn.ConvTranspose2d(f1*8, f1*4, 2, stride=2)
        self.dconv1 = conv2d_block(f1*8, f1*4)
        
        self.up2 = nn.ConvTranspose2d(f1*4, f1*2, 2, stride=2)
        self.dconv2 = conv2d_block(f1*4, f1*2)
        
        self.up3 = nn.ConvTranspose2d(f1*2, f1, 2, stride=2)
        self.dconv3 = conv2d_block(f1*2, f1)

        self.conv_last = nn.Conv2d(f1, 1, 1)

    def forward(self, enc_out, conv3, conv2, conv1):
        
        x_3 = self.up1(enc_out)
        x_3 = torch.cat((x_3, conv3), dim=1)
        conv_3 = self.dconv1(x_3)
        
        x_2 = self.up2(conv_3)
        x_2 = torch.cat((x_2, conv2), dim=1)
        conv_2 = self.dconv2(x_2)
        
        x_1 = self.up3(conv_2)
        x_1 = torch.cat((x_1, conv1), dim=1)
        conv_1 = self.dconv3(x_1)
        
        class_output = self.conv_last(conv_1)
        class_output = torch.sigmoid(class_output)
        
        return class_output   

class AtrousUNet(nn.Module):
    """
    Atrous UNet architecture

    Attributes:
        f1 (int): number of filters in the first convolutional layers
        depth (int): number of atrous convolutional layers
        mode (str): exists two modes, but just one was implemented in pytorch
        
    """
    def __init__(self, f1=16, depth=6, mode="cascade"):
        super(AtrousUNet, self).__init__()
        self.encoder = Encoder(f1, depth=depth, mode=mode)
        self.decoder = Decoder(f1)

    def forward(self, x):
        enc_out, conv3, conv2, conv1 = self.encoder(x)
        dec_out = self.decoder(enc_out, conv3, conv2, conv1)
        
        return dec_out
