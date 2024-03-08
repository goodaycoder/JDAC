from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, MaxPool3d, AvgPool1d, Dropout3d
from torch.nn import ReLU, Sigmoid
from functools import partial

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.basic_unet import TwoConv,Down,UpCat
#from src.models.condrefinenet3d import ConditionalInstanceNorm3dPlus
from monai.networks.nets import UNet,BasicUNet
import torch.nn.functional as F

class ConditionalInstanceNorm3dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features

        middle_feature = num_features//4
        self.gamma_conv = nn.Sequential(
            nn.Conv3d(num_features,middle_feature,1),
            nn.ReLU(True),
            nn.Conv3d(middle_feature,num_features,1),
        )
        self.alpha_conv = nn.Sequential(
            nn.Conv3d(num_features,middle_feature,1),
            nn.ReLU(True),
            nn.Conv3d(middle_feature,num_features,1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv3d(num_features,middle_feature,1),
            nn.ReLU(True),
            nn.Conv3d(middle_feature,num_features,1),
        )

        def weight_init_gamma(m):
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                m.weight.data.normal_(1/num_features, 0.02)
        def weight_init_beta(m):
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                m.weight.data.normal_(0, 0.02)

        self.gamma_conv.apply(weight_init_gamma)
        self.alpha_conv.apply(weight_init_gamma)
        self.beta_conv.apply(weight_init_beta)

        self.instance_norm = nn.InstanceNorm3d(num_features, affine=False, track_running_stats=False)

    def forward(self, x, y):
        y = y.repeat(1, x.shape[1], 1, 1, 1)
        means = torch.mean(x, dim=(2, 3, 4))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        means = means.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        y = y * means
        gamma, alpha, beta = self.gamma_conv(y), self.alpha_conv(y), self.beta_conv(y)
        h = h + means * alpha
        out = gamma * h + beta
        return out

class Denoiser_CondUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (16, 32, 64, 128, 128, 16),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.act  = nn.ELU(inplace=True)
        self.norm = ConditionalInstanceNorm3dPlus
        self.normalizer = self.norm(fea[5])

        
        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        #in_chns: number of input channels to be upsampled.           cat_chns: number of channels from the encoder.         out_chns: number of output channels.

        #self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_4 = UpCat(spatial_dims, fea[4], 0, fea[3], act, norm, bias, dropout, upsample)
        #self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], 0, fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv1= Conv["conv", spatial_dims](fea[1], out_channels, kernel_size=1)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        self.final_up   = UpCat(spatial_dims, fea[5], 0, out_channels, act, norm, bias, dropout, upsample, halves=False)
        self.final_down = Down(spatial_dims, out_channels, out_channels, act, norm, bias, dropout)


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        #u4 = self.upcat_4(x4, x3)
        u4 = self.upcat_4(x4, None)
        #u3 = self.upcat_3(u4, x2)
        u3 = self.upcat_3(u4, None)
        u2 = self.upcat_2(u3, x1)
        #u2 = self.upcat_2(u3)
        u1 = self.upcat_1(u2, x0)
        #u1 = self.upcat_1(u2)
        logits_down = self.final_conv1(u2)
        output = self.normalizer(u1, y)
        output = self.act(output)
        output = self.final_conv(output)

        return output, x - output, logits_down
    
    
class AntiART_UNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (16, 32, 64, 128, 128, 16),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        print(f"BasicUNet features: {fea}.")

        self.conv_0 = TwoConv(spatial_dims, in_channels, features[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        #in_chns: number of input channels to be upsampled.           cat_chns: number of channels from the encoder.         out_chns: number of output channels.

        #self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_4 = UpCat(spatial_dims, fea[4], 0, fea[3], act, norm, bias, dropout, upsample)
        #self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], 0, fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)
        self.final_conv1= Conv["conv", spatial_dims](fea[1], out_channels, kernel_size=1)

        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
        #self.final_up   = UpCat(spatial_dims, fea[5], 0, out_channels, act, norm, bias, dropout, upsample, halves=False)
        #self.final_down = Down(spatial_dims, out_channels, out_channels, act, norm, bias, dropout)


    def forward(self, x: torch.Tensor):
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        #u4 = self.upcat_4(x4, x3)
        u4 = self.upcat_4(x4, None)
        #u3 = self.upcat_3(u4, x2)
        u3 = self.upcat_3(u4, None)
        u2 = self.upcat_2(u3, x1)
        #u2 = self.upcat_2(u3)
        u1 = self.upcat_1(u2, x0)
        #u1 = self.upcat_1(u2)
        logits_down = self.final_conv1(u2)
        logits = self.final_conv(u1)
        return logits, logits_down
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride_x, stride_y):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(stride_x, stride_y), padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class MultiResolutionBlock(nn.Module):
    def __init__(self,in_ch, n_f):
        super(MultiResolutionBlock, self).__init__()
        self.conv_block0 = ConvBlock(in_ch, n_f, 1, 1)
        self.conv_block1 = ConvBlock(n_f, n_f, 1, 1)
        self.conv_block2 = ConvBlock(n_f*2, n_f, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

    def forward(self, x):
        x1 = self.conv_block0(x)
        x1 = self.conv_block1(x1)
        pool1 = self.pool(x1)
        
        x2 = self.conv_block1(pool1)
        x2 = self.conv_block1(x2)
        pool2 = self.pool(x2)
        
        x3 = self.conv_block1(pool2)
        x3 = self.conv_block1(x3)
        pool3 = self.pool(x3)
        
        x4 = self.conv_block1(pool3)
        x4 = self.conv_block1(x4)
        pool4 = self.pool(x4)

        x5 = self.conv_block1(pool4)
        x5 = self.conv_block1(x5)
    
    
        up6 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        up6 = self.conv_block1(up6)
        merge6 = torch.cat([x4,up6], dim=1)
        x6 = self.conv_block2(merge6)
        x6 = self.conv_block1(x6)
    
        up7 = F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=False)
        up7 = self.conv_block1(up7)
        merge7 = torch.cat([x3,up7], dim=1)#([x3,up7])
        x7 = self.conv_block2(merge7)
        x7 = self.conv_block1(x7)

        up8 = F.interpolate(x7, scale_factor=2, mode='bilinear', align_corners=False)
        up8 = self.conv_block1(up8)
        merge8 = torch.cat([x2,up8], dim=1)#([x2,up8])
        x8 = self.conv_block2(merge8)
        x8 = self.conv_block1(x8)

        up9 = F.interpolate(x8, scale_factor=2, mode='bilinear', align_corners=False)
        up9 = self.conv_block1(up9)
        merge9 = torch.cat([x1,up9], dim=1)#([x1,up9])
        x9 = self.conv_block2(merge9)
        x9 = self.conv_block1(x9)
        return x9

class DenseBlock(nn.Module):
    def __init__(self, num_blocks, n_f):
        super(DenseBlock, self).__init__()
        #self.startblock = MultiResolutionBlock(1,n_f)
        self.blocks = nn.ModuleList([MultiResolutionBlock(i*n_f+1,n_f) for i in range(num_blocks)])
        self.conv = ConvBlock((num_blocks)* n_f+1, 1, 1, 1)

    def forward(self, x):
        list_feat = [x]
        #x = self.startblock(x)
        #list_feat.append(x)
        for block in self.blocks:
            x = block(x)
            list_feat.append(x)
            x = torch.cat(list_feat, dim=1)
        return self.conv(x)

class DRN_DCMB(nn.Module):
    def __init__(self, num_dense_blocks, n_f):
        super(DRN_DCMB, self).__init__()
        self.dense_block = DenseBlock(num_dense_blocks, n_f)
        #self.subtract = nn.quantized.FloatFunctional()
        #self.conv = ConvBlock((num_blocks)* n_f+1, 1, 1, 1)

    def forward(self, x):
        x_b = self.dense_block(x)
        #x = self.conv(x)
        return torch.sub(x, x_b)
    
# Assuming input size is (batch_size, channels, height, width)
#input_size = (1, 1, None, None)
#model = DRN_DCMB(3, 32)    
    
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(CBAMBlock, self).__init__()

        self.channel_attention = self._make_channel_attention(in_channels, ratio)
        self.spatial_attention = self._make_spatial_attention(in_channels)

    def forward(self, x):
        x_ca = self.channel_attention(x)
        x_ca = torch.mul(x, x_ca)
        x_sa = self.spatial_attention(x_ca)
        return torch.mul(x, x_sa)

    def _make_channel_attention(self, in_channels, ratio):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def _make_spatial_attention(self, in_channels):
        kernel_size = 7
        return nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.cbam = CBAMBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.cbam(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.block1 = UNetBlock(in_channels, 32)
        self.block2 = UNetBlock(32, 64)
        self.block3 = UNetBlock(64, 128)
        self.block4 = UNetBlock(128, 256)

        self.up1 = UNetBlock(256+128, 128)
        self.up2 = UNetBlock(128+64, 64)
        self.up3 = UNetBlock(64+32, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.block1(x)#128*128
        x = F.avg_pool2d(x1, kernel_size=2, stride=2)

        x2 = self.block2(x)#64*64
        x = F.avg_pool2d(x2, kernel_size=2, stride=2)

        x3 = self.block3(x)#32*32
        x = F.avg_pool2d(x3, kernel_size=2, stride=2)

        x = self.block4(x)#16*16

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)#32*32
        x = self.up1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)#64*64
        x = self.up2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)#128*128
        x = self.up3(x)

        x = self.final_conv(x)

        return x

class CorrectionMultiInput(nn.Module):
    def __init__(self, in_channels, out_channels, input_height, input_width):
        super(CorrectionMultiInput, self).__init__()

        assert input_height % 32 == 0
        assert input_width % 32 == 0

        self.conv_initial = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(32)
        self.relu_initial = nn.ReLU(inplace=True)
        self.unet1 = UNet(32*3, out_channels)
        self.unet2 = UNet(32*3+out_channels, out_channels)

    def forward(self, x):
        x1,x2,x3 = torch.split(x, 1, dim=1)
        x1 = self.conv_initial(x1)
        x1 = self.bn_initial(x1)
        x1 = self.relu_initial(x1)
        x2 = self.conv_initial(x2)
        x2 = self.bn_initial(x2)
        x2 = self.relu_initial(x2)
        x3 = self.conv_initial(x3)
        x3 = self.bn_initial(x3)
        x3 = self.relu_initial(x3)
        input_concat = torch.cat([x1, x2, x3], dim=1)
        ## Two Stacked Nets:
        pred_1  = self.unet1(input_concat)
        input_2 = torch.cat([input_concat, pred_1], dim=1) 
        pred_2  = self.unet2(input_2)        
        return pred_2

# Example usage:
#input_height = 256
#input_width = 256
#in_channels = 1
#out_channels = 1
#model = CorrectionMultiInput(in_channels, out_channels, input_height, input_width)