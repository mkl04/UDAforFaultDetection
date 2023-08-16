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
    def __init__(self, f1=16):
        super(Encoder, self).__init__()
        
        self.dconv1 = conv2d_block(1, f1)
        self.dconv2 = conv2d_block(f1, f1*2)
        self.dconv3 = conv2d_block(f1*2, f1*4)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        # self.dropout = nn.Dropout()

        self.bottle = conv2d_block(f1*4, f1*8)

    def forward(self, x):

        conv1 = self.dconv1(x)
        x1 = self.maxpool(conv1)

        conv2 = self.dconv2(x1)
        x2 = self.maxpool(conv2)
        
        conv3 = self.dconv3(x2)
        x3 = self.maxpool(conv3)
        # x3 = self.dropout(x3)
        
        enc_out = self.bottle(x3)       
        
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
        
        return class_output, conv_1, conv_2, conv_3 

class UNet2D(nn.Module):
    """
    UNet architecture for DANN

    Attributes:
        f1 (int): number of filters in the first convolutional layers
        branches (list): list of branches to compute DANN
        
    """
    def __init__(self, f1=16, branches=['conv_3']):
        super(UNet2D, self).__init__()
        self.encoder = Encoder(f1)
        self.decoder = Decoder(f1)

        self.branches = branches

        if 'conv_3' in branches:
        
            # 64, 32x32
            self.conv_d3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            
            self.discr_d3 = nn.Sequential(
                nn.Linear(32*16*16, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 1),
                nn.Sigmoid()
            )

    def forward(self, x, alpha):
        enc_out, conv3, conv2, conv1 = self.encoder(x)
        dec_out, conv_1, conv_2, conv_3 = self.decoder(enc_out, conv3, conv2, conv1)
        
        for branch_name in self.branches:
        
            if 'conv_3' in branch_name:
                branch = self.conv_d3(conv_3)
                branch = self.maxpool(branch)

                features = branch.view(branch.size(0), -1) # Flatten
                reverse_feature = ReverseLayerF.apply(features, alpha)
                domain_out = self.discr_d3(reverse_feature)
                
        return dec_out, domain_out
    

# For GRL

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None