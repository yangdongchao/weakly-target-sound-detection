# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
from itertools import zip_longest
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import math
DEBUG=0
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class MaxPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.max(decision, dim=self.pooldim)[0]

class nnMeanAndMax(nn.Module):
    def __init__(self):
        super(nnMeanAndMax,self).__init__()
    def forward(self, x):
        x = torch.mean(x,dim=-1)
        (x, _) = torch.max(x,dim=-1)
        return x

class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / (time_decision.sum(
            self.pooldim)+1e-7)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527):
        
        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 128, bias=True)
        self.fc_audioset = nn.Linear(128, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input_, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        input_ = input_.unsqueeze(1)
        x = self.conv_block1(input_, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        # print(x.shape)
        # x = torch.mean(x, dim=3)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = self.fc1(x)
        # print(x.shape)
        # assert 1==2
        # (x1,_) = torch.max(x, dim=2)
        # x2 = torch.mean(x, dim=2)
        # x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        return x

class Cnn10(nn.Module):
    def __init__(self):
        
        super(Cnn10, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        # self.fc1 = nn.Linear(512, 512, bias=True)
        # self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        # self.init_weight()
 
    def forward(self, input, fi=None):
        """
        Input: (batch_size, data_length)"""

        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma+1)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma+1)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma+1)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 4), pool_type='avg')
        if fi != None:
            gamma = fi[:,0].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            beta = fi[:,1].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
            x = (gamma+1)*x + beta
        x = F.dropout(x, p=0.2, training=self.training)
        return x

class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class ResPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.linPool = LinearSoftPool(pooldim=1)

class AutoExpPool(nn.Module):
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.full((outputdim, ), 1))
        self.pooldim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        return (logits * torch.exp(scaled)).sum(
            self.pooldim) / torch.exp(scaled).sum(self.pooldim)


class SoftPool(nn.Module):
    def __init__(self, T=1, pooldim=1):
        super().__init__()
        self.pooldim = pooldim
        self.T = T

    def forward(self, logits, decision):
        w = torch.softmax(decision / self.T, dim=self.pooldim)
        return torch.sum(decision * w, dim=self.pooldim)


class AutoPool(nn.Module):
    """docstring for AutoPool"""
    def __init__(self, outputdim=10, pooldim=1):
        super().__init__()
        self.outputdim = outputdim
        self.alpha = nn.Parameter(torch.ones(outputdim))
        self.dim = pooldim

    def forward(self, logits, decision):
        scaled = self.alpha * decision  # \alpha * P(Y|x) in the paper
        weight = torch.softmax(scaled, dim=self.dim)
        return torch.sum(decision * weight, dim=self.dim)  # B x C


class ExtAttentionPool(nn.Module):
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.attention = nn.Linear(inputdim, outputdim)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)
        self.activ = nn.Softmax(dim=self.pooldim)

    def forward(self, logits, decision):
        # Logits of shape (B, T, D), decision of shape (B, T, C)
        w_x = self.activ(self.attention(logits) / self.outputdim)
        h = (logits.permute(0, 2, 1).contiguous().unsqueeze(-2) *
             w_x.unsqueeze(-1)).flatten(-2).contiguous()
        return torch.sum(h, self.pooldim)


class AttentionPool(nn.Module):
    """docstring for AttentionPool"""
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):
        super().__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.pooldim = pooldim
        self.transform = nn.Linear(inputdim, outputdim)
        self.activ = nn.Softmax(dim=self.pooldim)
        self.eps = 1e-7

    def forward(self, logits, decision):
        # Input is (B, T, D)
        # B, T , D
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))
        detect = (decision * w).sum(
            self.pooldim) / (w.sum(self.pooldim) + self.eps)
        # B, T, D
        return detect

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))
    def forward(self, x):
        return self.block(x)

class AudioCNN(nn.Module):
    def __init__(self, classes_num):
        super(AudioCNN, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512,128,bias=True)
        self.fc = nn.Linear(128, classes_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        # [128, 801, 168] --> [128,1,801,168]
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # 128,64,400,84
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg') # 128,128,200,42
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg') # 128,256,100,21
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') # 128,512,50,10
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes) # 128,512,50
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps) 128,512
        x = self.fc1(x) # 128,128
        output = self.fc(x) # 128,10
        return x,output

    def extract(self,input):
        '''Input: (batch_size, times_steps, freq_bins)'''
        x = input[:, None, :, :]
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc1(x) # 128,128
        return x

def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1

    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'max':
        return MaxPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'expalpha':
        return AutoExpPool(outputdim=kwargs['outputdim'], pooldim=1)

    elif poolingfunction_name == 'soft':
        return SoftPool(pooldim=1)
    elif poolingfunction_name == 'auto':
        return AutoPool(outputdim=kwargs['outputdim'])
    elif poolingfunction_name == 'attention':
        return AttentionPool(inputdim=kwargs['inputdim'],
                             outputdim=kwargs['outputdim'])
class conv1d(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding='VALID', dilation=1):
        super(conv1d, self).__init__()
        if padding == 'VALID':
            dconv_pad = 0
        elif padding == 'SAME':
            dconv_pad = dilation * ((kernel_size - 1) // 2)
        else:
            raise ValueError("Padding Mode Error!")
        self.conv = nn.Conv1d(nin, nout, kernel_size=kernel_size, stride=stride, padding=dconv_pad)
        self.act = nn.ReLU()
        self.init_layer(self.conv)

    def init_layer(self, layer, nonlinearity='relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
        nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        out = self.act(self.conv(x))
        return out

class Atten_1(nn.Module):
    def __init__(self, input_dim, context=2, dropout_rate=0.2):
        super(Atten_1, self).__init__()
        self._matrix_k = nn.Linear(input_dim, input_dim // 4)
        self._matrix_q = nn.Linear(input_dim, input_dim // 4)
        self.relu = nn.ReLU()
        self.context = context
        self._dropout_layer = nn.Dropout(dropout_rate)
        self.init_layer(self._matrix_k)
        self.init_layer(self._matrix_q)

    def init_layer(self, layer, nonlinearity='leaky_relu'):
        """Initialize a Linear or Convolutional layer. """
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

    def forward(self, input_x):
        k_x = input_x
        k_x = self.relu(self._matrix_k(k_x))
        k_x = self._dropout_layer(k_x)
        # print('k_x ',k_x.shape)
        q_x = input_x[:, self.context, :]
        # print('q_x ',q_x.shape)
        q_x = q_x[:, None, :]
        # print('q_x1 ',q_x.shape)
        q_x = self.relu(self._matrix_q(q_x))
        q_x = self._dropout_layer(q_x)
        # print('q_x2 ',q_x.shape)
        x_ = torch.matmul(k_x, q_x.transpose(-2, -1) / math.sqrt(k_x.size(-1)))
        # print('x_ ',x_.shape)
        x_ = x_.squeeze(2)
        alpha = F.softmax(x_, dim=-1)
        att_ = alpha
        # print('alpha ',alpha)
        alpha = alpha.unsqueeze(2).repeat(1,1,input_x.shape[2])
        # print('alpha ',alpha)
        # alpha = alpha.view(alpha.size(0), alpha.size(1), alpha.size(2), 1)
        out = alpha * input_x
        # print('out ', out.shape)
        # out = out.mean(2)
        out = out.mean(1)
        # print('out ',out.shape)
        # assert 1==2
        #y = alpha * input_x
        #return y, att_
        return out

class Fusion(nn.Module):
    def __init__(self, inputdim,inputdim2,n_fac):
        super().__init__()
        self.fuse_layer1 = conv1d(inputdim, inputdim2*n_fac,1)
        self.fuse_layer2 = conv1d(inputdim2, inputdim2*n_fac,1)
        self.avg_pool = nn.AvgPool1d(n_fac, stride=n_fac) # 沿着最后一个维度进行pooling

    def forward(self,embedding,mix_embed):
        embedding = embedding.permute(0,2,1)
        fuse1_out = self.fuse_layer1(embedding) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse1_out = fuse1_out.permute(0,2,1)

        mix_embed = mix_embed.permute(0,2,1)
        fuse2_out = self.fuse_layer2(mix_embed) # [2, 501, 2560] ,512*5, 1D卷积融合,spk_embeding ,扩大其维度 
        fuse2_out = fuse2_out.permute(0,2,1)
        as_embs = torch.mul(fuse1_out, fuse2_out) # 相乘 [2, 501, 2560]
        # (10, 501, 512)
        as_embs = self.avg_pool(as_embs) # [2, 501, 512] 相当于 2560//5
        return as_embs

class CDur_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.fusion = Fusion(128,2)
        self.fc = nn.Linear(256,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding,x)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(256, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_big(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 2)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (2, 4)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),)
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

class CDur_CNN14(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up

class CDur_CNN14_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.fusion = Fusion(128,512,2)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding,x)
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up

class CDur_CNN14_my(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        # self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        # if not hasattr(self, '_flattened'):
        #     self.gru.flatten_parameters()
        # x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up

class Join(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14(inputdim,outputdim,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def embedding_enhancement(self,x,embedding):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,2)
        embedding_tmp = embedding.unsqueeze(1)
        att_w = self.get_w(embedding_tmp,selected_embeddings)
        mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        # att = top_k-0.55
        # # print('att ',att)
        # att_0 = top_k > 0.55
        # # print('att_0 ',att_0)
        # att = att_0 * att
        # # print('att ',att.shape)
        # att = att.unsqueeze(1)
        # att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        if top_k.all() < 0.55:
            final_embedding = embedding
        else:
            final_embedding = embedding + 0.1*mix_embedding
        return final_embedding

    def forward(self,x,ref,att_pool=False,enhancement=False):
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if enhancement == True:
            embedding = self.embedding_enhancement(x,embedding)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up,logit

class Join_clr(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14(inputdim,outputdim,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        self.r_easy = 10 # easy 数量？
        self.r_hard = 20 # hard 数量?
        self.m = 3
        self.M = 6
        self.dropout = nn.Dropout(p=0.3)
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        # print('_ ',_[0])
        top_k = _[:,:k]
        # print('top_k ',top_k[0])
        top_k = top_k.mean(1)
        # print('top_k ',top_k[0])
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def embedding_enhancement(self,x,embedding):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        # print('mixture_embedding ',mixture_embedding.shape)
        # assert 1==2
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,5)
        # print('selected_embeddings ',selected_embeddings.shape)
        # print('embedding ',embedding.shape)
        embedding_tmp = embedding.unsqueeze(1)
        #assert 1==2
        att_w = self.get_w(embedding_tmp,selected_embeddings)
        # print('att_w ',att_w[0])
        mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        att = top_k-0.55
        # print('att ',att)
        att_0 = top_k > 0.55
        # print('att_0 ',att_0)
        att = att_0 * att
        # print('att ',att.shape)
        att = att.unsqueeze(1)
        att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        final_embedding = embedding + att*mix_embedding
        # print('final_embedding ',final_embedding.shape)
        # assert 1==2
        return final_embedding

    def select_topk_embeddings_clr(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,idx_topk

    def select_topk_embeddings_hard(self, scores, embeddings, k, hard_index):
        # print('embeddings ',embeddings.shape)
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk_tmp = idx_topk.cpu().detach().numpy()
        ans = []
        for i,idx in enumerate(idx_topk_tmp):
            tmp = np.intersect1d(idx,np.array(hard_index[i])).tolist()
            if len(tmp)==0:
                tmp = hard_index[i]
            if len(tmp) < k:
                for t in hard_index[i]:
                    tmp.append(t)
                    if len(tmp) >= k:
                        break
            if len(tmp) < k:
                for t in idx_topk_tmp[i]:
                    tmp.append(t)
                    if len(tmp) >= k:
                        break
            # print(len(tmp))
            ans.append(torch.tensor(tmp[:k]).cuda())
        idx_topk = torch.stack(ans,0).cuda()
        # print(idx_topk.shape)
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,idx_topk
    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)
        actionness_drop = actionness * select_idx
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx
        easy_act,easy_act_id = self.select_topk_embeddings_clr(actionness_drop, embeddings, k_easy)
        easy_bkg,easy_bkg_id = self.select_topk_embeddings_clr(actionness_rev_drop, embeddings, k_easy)
        return easy_act,easy_act_id, easy_bkg,easy_bkg_id

    def hard_snippets_mining(self, actionness, embeddings, k_hard,labels=None):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        #aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)
        aness_bin = labels.cpu().detach().numpy()
        hard_act_index, hard_bkg_index = self.hard_frame_from_label(actionness,labels)
        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act,hard_act_id = self.select_topk_embeddings_hard(aness_region_inner, embeddings, k_hard,hard_act_index)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg,hard_bkg_id = self.select_topk_embeddings_hard(aness_region_outer, embeddings, k_hard,hard_bkg_index)
        return hard_act,hard_act_id, hard_bkg,hard_bkg_id

    def hard_frame_from_label(self,predict,labels):
        predict_np = predict.cpu().detach().numpy()
        predict_np = predict_np > 0.5
        labels_np = labels.cpu().detach().numpy()
        labels_np = labels_np > 0.5
        change_index = []
        for i in range(predict_np.shape[0]):
            ls = np.logical_xor(predict_np[i], labels_np[i]).nonzero()[0]
            change_index.append(ls)
        hard_act_index = []
        hard_bkg_index = []
        for i,line in enumerate(change_index):
            tmp = []
            tmp2 = []
            for t in line:
                if labels_np[i,t] == True:
                    tmp.append(t)
                else:
                    tmp2.append(t)
            hard_act_index.append(tmp)
            hard_bkg_index.append(tmp2)
        return hard_act_index, hard_bkg_index
    
    def choose_sequence(self,easy_act_id,easy_bkg_id,hard_act_id,hard_bkg_id):
        easy_act_dict = {}
        easy_bkg_dict = {}
        hard_act_dict = {}
        hard_bkg_dict = {}
        k = 0
        for ea in easy_act_id:
            easy_act_dict[ea[0]] = k
            k += 1

        k = 0
        for ea in easy_bkg_id:
            easy_bkg_dict[ea[0]] = k
            k += 1
        
        k = 0
        for ea in hard_act_id:
            hard_act_dict[ea[0]] = k
            k += 1

        k = 0
        for ea in hard_bkg_id:
            hard_bkg_dict[ea[0]] = k
            k += 1

        st_easy_act_dict = sorted(easy_act_dict)
        st_easy_bkg_dict = sorted(easy_bkg_dict)
        st_hard_act_dict = sorted(hard_act_dict)
        st_hard_bkg_dict = sorted(hard_bkg_dict)
        answer_dict = {}
        for hard_ac in st_hard_act_dict:
            id = hard_ac[0]
            for easy_ac in st_easy_act_dict:
                if easy_ac[0] > id:
                    if id not in answer_dict.keys():
                        answer_dict[id] = [easy_ac[0]]
                    else:
                        answer_dict[id].append(easy_ac[0])
            

        print('easy_act_id ',easy_act_id[0])
        print('easy_bkg_id ',easy_bkg_id[0])
        print('hard_act_id ',hard_act_id[0])
        print('hard_bkg_id ',hard_bkg_id[0])
        assert 1==2
        pass

    def clr(self,predict,labels):
        predict_np = predict.cpu().detach().numpy()
        predict_np = predict_np > 0.5
        labels_np = labels.cpu().detach().numpy()
        labels_np = labels_np > 0.5
        change_indices = np.logical_xor(predict_np, labels_np).nonzero() # .nonzero()[0]


    def forward(self,x,ref,labels=None,att_pool=False,enhancement=False):
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if enhancement == True:
            embedding = self.embedding_enhancement(x,embedding)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        clr_embeddings = x
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard # k_easy, k_hard 25 6
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        if labels == None:
            contrast_pairs = {
            'EA': clr_embeddings[:,0],
            'EB': clr_embeddings[:,0],
            'HA': clr_embeddings[:,0],
            'HB': clr_embeddings[:,0]}
        else:
            easy_act,easy_act_id, easy_bkg,easy_bkg_id = self.easy_snippets_mining(decision_time[:,:,0], clr_embeddings, k_easy)
            hard_act,hard_act_id, hard_bkg,hard_bkg_id = self.hard_snippets_mining(decision_time[:,:,0], clr_embeddings, k_hard,labels)
            # self.choose_sequence(easy_act_id,easy_bkg_id,hard_act_id,hard_bkg_id)
            contrast_pairs = {
                'EA': easy_act,
                'EB': easy_bkg,
                'HA': hard_act,
                'HB': hard_bkg}
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,logit,contrast_pairs

class Join_fusion(nn.Module):
    def __init__(self,model_config,inputdim,outputdim,**kwargs):
        super().__init__()
        self.encoder = Cnn14()
        self.detection = CDur_CNN14_fusion(inputdim,outputdim,**kwargs)
        self.softmax = nn.Softmax(dim=2)
        self.temperature = 5
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        if model_config['pre_train']:
            self.encoder.load_state_dict(torch.load(model_config['encoder_path'])['model'])
            self.detection.load_state_dict(torch.load(model_config['CDur_path']))
    
    def get_w(self,q,k):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn/self.temperature
        attn = self.softmax(attn)
        return attn
    
    def attention_pooling(self,embeddings,mean_embedding):
        att_pool_w = self.get_w(mean_embedding,embeddings)
        embedding = torch.bmm(att_pool_w, embeddings).squeeze(1)
        # print(embedding.shape)
        # print(att_pool_w.shape)
        # print(att_pool_w[0])
        # assert 1==2
        return embedding
    
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k
    
    def embedding_enhancement(self,x,embedding):
        batch, time, dim = x.shape
        mixture_embedding = self.encoder(x) # 8, 125, 128
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding_pre = embedding.unsqueeze(1)
        embedding_pre = embedding_pre.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding_pre), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        selected_embeddings,top_k = self.select_topk_embeddings(decision_time[:,:,0],mixture_embedding,2)
        embedding_tmp = embedding.unsqueeze(1)
        att_w = self.get_w(embedding_tmp,selected_embeddings)
        mix_embedding = torch.bmm(att_w, selected_embeddings).squeeze(1)
        # print('mix_embedding ',mix_embedding.shape)
        # assert 1==2
        # att = top_k-0.55
        # # print('att ',att)
        # att_0 = top_k > 0.55
        # # print('att_0 ',att_0)
        # att = att_0 * att
        # # print('att ',att.shape)
        # att = att.unsqueeze(1)
        # att = att.repeat(1,mix_embedding.shape[1])
        # print(att.shape)
        # print(att[0])
        # assert 1 == 2
        if top_k < 0.55:
            final_embedding = embedding
        else:
            final_embedding = embedding + 0.1*mix_embedding
        return final_embedding

    def forward(self,x,ref,att_pool=False,enhancement=False):
        logit = torch.zeros(1).cuda()
        embeddings  = self.encoder(ref)
        mean_embedding = embeddings.mean(1).unsqueeze(1)
        if att_pool == True:
            embedding = self.attention_pooling(embeddings,mean_embedding)
        else:
            embedding = mean_embedding.squeeze(1)
        if enhancement == True:
            embedding = self.embedding_enhancement(x,embedding)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.detection.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        x = self.detection.fusion(embedding,x) 
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.detection.gru.flatten_parameters()
        x, _ = self.detection.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.detection.fc(x)
        decision_time = torch.softmax(self.detection.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up,logit
    
class AudioNet(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 2)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (1, 4)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(1024,512)
        self.outputlayer = nn.Linear(512, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # (b,512,125,1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        #x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7,1.) # x  torch.Size([16, 125, 369])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time,decision_up

class AudioNet2(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,128)
        # embedding = embedding.unsqueeze(1)
        # embedding = embedding.repeat(1, x.shape[1], 1)
        # x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7,1.) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time, decision_up

class LSTMCell(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        #gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)

class LSTMCell2(nn.Module):
    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.x2h_f = nn.Linear(128, hidden_size, bias=bias)
        self.h2h_f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    def forward(self, x, hidden):
        # print('x ',x.shape)
        # print('hidden ',hidden[0].shape,hidden[1].shape)
        embedding = x[:,-128:]
        # print('embdding ',embedding.shape)
        # assert 1==2
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        gates = self.x2h(x) + self.h2h(hx)
        # print('gates ',gates.shape)
        # gates = gates.squeeze()
        # #if 
        # print('gates ',gates.shape)
        # assert 1==2
        ingate , cellgate, outgate = gates.chunk(3, 1)
        # print('embedding ',embedding.shape)
        # print('hx ',hx.shape)
        forgetgate = self.x2h_f(embedding) + self.h2h_f(hx)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = torch.mul(cx, forgetgate) +  torch.mul(ingate, cellgate)        
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.lstm = LSTMCell(input_dim, hidden_dim, layer_dim)
        self.lstm_r = LSTMCell(input_dim, hidden_dim, layer_dim)  
        # self.fc = nn.Linear(hidden_dim, output_dim)
        self.att = Atten_1(input_dim)
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_h0 =  torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            rever_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        cn_r = rever_c0[0,:,:]
        hn_r = rever_h0[0,:,:]
        for seq in range(x.size(1)):
            if seq > 2:
                input_x = x[:,seq-2:min(seq+3,x.size(1)),:]
                x_tmp = self.att(input_x)
            else:
                x_tmp = x[:,seq,:]
            reverse_seq = x.size(1)-seq-1
            if reverse_seq < x.size(1)-2:
                input_x = x[:,max(0,reverse_seq-2):reverse_seq+3,:]
                x_tmp_reverse = self.att(input_x)
            else:
                x_tmp_reverse = x[:,reverse_seq,:]
            hn, cn = self.lstm(x_tmp, (hn,cn)) 
            hn_r,cn_r = self.lstm_r(x_tmp_reverse,(hn_r,cn_r))
            # print('hn_r, hn, cn',hn_r.shape,hn.shape,cn.shape)
            # print(torch.cat((hn,hn_r),1).shape)
            # assert 1==2
            outs.append(torch.cat((hn,hn_r),1))
        outs = torch.stack(outs,1)
        # print('out ',outs.shape)
        # out = self.fc(out) 
        return outs

class LSTMModel2(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(LSTMModel2, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        self.lstm = LSTMCell2(input_dim, hidden_dim, layer_dim)
        self.lstm_r = LSTMCell2(input_dim, hidden_dim, layer_dim)  
        # self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_h0 =  torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            rever_h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # Initialize cell state
        if torch.cuda.is_available():
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        else:
            c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
            rever_c0 = torch.zeros(self.layer_dim, x.size(0), hidden_dim)
        outs = []
        cn = c0[0,:,:]
        hn = h0[0,:,:]
        cn_r = rever_c0[0,:,:]
        hn_r = rever_h0[0,:,:]
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:,seq,:], (hn,cn)) 
            hn_r,cn_r = self.lstm_r(x[:,x.size(1)-seq-1,:],(hn_r,cn_r))
            # print('hn_r, hn, cn',hn_r.shape,hn.shape,cn.shape)
            # print(torch.cat((hn,hn_r),1).shape)
            # assert 1==2
            outs.append(torch.cat((hn,hn_r),1))
        outs = torch.stack(outs,1)
        # print('out ',outs.shape)
        # out = self.fc(out) 
        return outs

class FilML_generator2(nn.Module):
    def __init__(self,embedding_dim,lstm_hidden_dim_q):
        super(FilML_generator2,self).__init__()
        self.lstm_q = nn.Linear(embedding_dim,lstm_hidden_dim_q)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(lstm_hidden_dim_q,2)
        # self.fc2 = nn.Linear(time_dim,4)

    def forward(self,x):
        embeddings = self.lstm_q(x)
        embeddings = self.relu(embeddings)
        # embeddings = embeddings.view(embeddings.shape[0],4,-1)
        # print('embeddings ',embeddings.shape)
        embeddings = self.fc1(embeddings)
        return embeddings

class S_student(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up

    def extract(self,x, embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        out = x
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        return out,decision_time[:,0]
        
class S_student_mask(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(642, 512, bidirectional=True, batch_first=True)
        self.mask_generator = MaskGenerator()
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding, w_label=None): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        mask_loss, masks = self.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,514)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + 514] = 642
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up, mask_loss

    def extract(self,x, embedding, w_label=None):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        mask_loss, masks = self.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        out = x
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        return out, decision_time[:,0]

class W_student_fusion(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        self.gru = nn.GRU(512, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.fusion = Fusion(128,512,2)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = self.fusion(embedding,x)
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up


class W_student(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(640, 512, bidirectional=True, batch_first=True)
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up

    def extract(self,x,embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        sim_embeddings = x
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        return decision_time[:,:,0], sim_embeddings
    
class W_student_mask(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = Cnn10()
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]
        self.gru = nn.GRU(642, 512, bidirectional=True, batch_first=True)
        self.mask_generator = MaskGenerator()
        # self.gru = LSTMModel(640, 512,1)
        self.fc = nn.Linear(1024,256)
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),inputdim=256,outputdim=outputdim)
        # self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)
    def forward(self, x, embedding, w_label): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        mask_loss, masks = self.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        clip_decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return clip_decision, decision_up, mask_loss

    def extract(self, x, embedding, w_label):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.features(x) # 
        mask_loss, masks = self.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.gru.flatten_parameters()
        x, _ = self.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.fc(x)
        sim_embeddings = x
        decision_time = torch.softmax(self.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        return decision_time[:,:,0], sim_embeddings

class Stage1(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        # self.fusion_s = nn.Linear(1024,256)
        # self.fusion_w = nn.Linear(1024,256)
    def forward(self,x,embedding):
        # the target is to train w_student
        s_out,s_decision_time = self.s_student.extract(x,embedding)
        #s_out = self.fusion_s(s_out)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        w_out = x
        #w_out = self.fusion_w(w_out)
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time

class Stage1_mask(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student_mask(inputdim, outputdim)
        self.w_student = W_student_mask(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        # self.fusion_s = nn.Linear(1024,256)
        # self.fusion_w = nn.Linear(1024,256)
    def forward(self, x, embedding, w_label):
        # the target is to train w_student
        s_out, s_decision_time = self.s_student.extract(x, embedding, w_label)
        #s_out = self.fusion_s(s_out)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        mask_loss, masks = self.w_student.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        w_out = x
        #w_out = self.fusion_w(w_out)
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time, w_clip_decision, w_decision_up, w_out, s_out, s_decision_time, mask_loss


class Stage2(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage1_path']) # we only need W_student part
        # print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
    
    def forward(self,x,embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1) # 
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up

class Stage2_from_s(nn.Module): # 我直接用s_student的参数来初始化w_student，然后再进行训练?
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage1_path']) # we only need W_student part
        # print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='s_student' and k[10:] in model_dict.keys()}
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
    
    def forward(self,x,embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1) # 
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up


class Stage2_mask(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.w_student = W_student_mask(inputdim, outputdim)
        save_model = torch.load(model_config['stage1_path']) # we only need W_student part
        # print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
    
    def forward(self,x,embedding,w_label):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        mask_loss, masks = self.w_student.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1) # 
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time, w_clip_decision, w_decision_up, mask_loss

class Stage3(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        # print('new....')
        # model_dict =  self.s_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        # assert 1==2
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        state_dict = {k[10:]:v for k,v in save_model.items() if (k[:9]=='w_student') and (k[10:] in model_dict.keys())}
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label


class Stage3_scrath(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        # print('new....')
        # model_dict =  self.s_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        # assert 1==2
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        state_dict = {k[10:]:v for k,v in save_model.items() if (k[:9]=='w_student') and (k[10:] in model_dict.keys())}
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label


class Stage3_mask(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student_mask(inputdim, outputdim)
        self.w_student = W_student_mask(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        # print('new....')
        # model_dict =  self.s_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        # assert 1==2
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        state_dict = {k[10:]:v for k,v in save_model.items() if (k[:9]=='w_student') and (k[10:] in model_dict.keys())}
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
    def forward(self, x, embedding, w_label):
        with torch.no_grad():
            pseudo_label, _ = self.w_student.extract(x, embedding, w_label) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        mask_loss, masks = self.s_student.mask_generator(x, embedding, w_label)
        x = torch.cat([x, masks], 1)
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, pseudo_label, mask_loss

class Stage4(nn.Module): # stage3 with simNet
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        self.simNet = SimilarityNet(256)
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[10:] in model_dict.keys()}
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        # init simNet
        simNet_save = torch.load(model_config['simnet_path'])
        # print('simNet_save ',simNet_save.keys())
        model_dict_sim =  self.simNet.state_dict()
        # for key in model_dict_sim.keys():
        #     print(model_dict_sim[key][0])
        #     break
        state_dict_sim = {k[7:]:v for k,v in simNet_save.items() if k[7:] in model_dict_sim.keys()}
        # print(state_dict_sim.keys())
        #assert 1==2
        model_dict_sim.update(state_dict_sim)
        #assert 1==2
        self.simNet.load_state_dict(model_dict_sim)
        #model_dict_sim =  self.simNet.state_dict()
        # print('after update....')
        # for key in model_dict_sim.keys():
        #     print(model_dict_sim[key][0])
        #     assert 1==2
        for p in self.simNet.parameters(): # fix the parameter of simNet
            p.requires_grad = False
        
    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1) # 根据分数进行排序
        top_k = _[:,:k]
        # top_k = top_k.mean(1)
        idx_topk = idx_DESC[:, :k] # 取top_k个
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings,top_k,idx_DESC[:, :k]
    
    def check_by_simnet(self,pseudo_label,embeddings):
        batch,frame_num = pseudo_label.shape
        # print('pseudo_label ',pseudo_label.shape)
        # print('embeddings ',embeddings.shape)
        # assert 1==2
        # selected_embeddings,top_k,idx = self.select_topk_embeddings(pseudo_label,embeddings,self.top_k)
        for i in range(batch):
            tmp = pseudo_label[i,:]
            # print(tmp)
            tmp_index = tmp > 0.7
            ans = tmp_index.nonzero()
            # print('ans ',ans.shape)
            ans = ans.squeeze()
            # print('ans ',ans)
            # assert 1==2
            if ans.shape[0] <= 1:
                continue
            selected_embeddings = embeddings[i,ans,:] # choose embed
            # print('selected_embeddings[] ',selected_embeddings[:,0])
            print('selected_embeddings ',selected_embeddings.shape)
            sim_ls = []
            for j in range(selected_embeddings.shape[0]):
                q = selected_embeddings[j,:] # 
                # print('q[0] ',q[0:10])
                # print('q ',q.shape)
                q = q.unsqueeze(0)
                # print('q1 ',q.shape)
                q = q.repeat(selected_embeddings.shape[0],1)
                # print('q ',q.shape)
                q_k = torch.cat([q,selected_embeddings],dim=1)
                # print('q_k[0] ',q_k[0,256:])
                print('q_k ',q_k.shape)
                with torch.no_grad():
                    scores = self.simNet(q_k)
                    print('scores ',scores.shape)
                scores = scores.squeeze()
                print('scores ',scores.shape)
                # print('scores ',scores)
                avg_score = scores.mean(0)
                print('avg_score ',avg_score)
                # assert 1==2
                sim_ls.append(avg_score)
            sim_ls = torch.stack(sim_ls,0)
            print('sim_ls ',sim_ls.shape)
            print(sim_ls)
            assert 1==2
            pseudo_label[i,ans] = pseudo_label[i,ans] * sim_ls # scale the predict results
        return pseudo_label

    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,sim_embeddings = self.w_student.extract(x,embedding) # get pseudo label
            pseudo_label = self.check_by_simnet(pseudo_label,sim_embeddings)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label

class Stage5(nn.Module):
    # mutual learning part
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        # 需要加载 stage3中训练得到的s_student的参数，然后加载stage2中的w_student 继续训练
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict) # 加载stage2中的w_student

        self.s_student = S_student(inputdim, outputdim)
        save_model_s = torch.load(model_config['stage3_path']) # we only need W_student part
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if k[:9]=='s_student' and k[10:] in model_dict_s.keys()}
        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s) # 加载stage2中的w_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        
    def forward(self,x,embedding):
        # the target is to train w_student
        s_out,s_decision_time = self.s_student.extract(x,embedding)
        #s_out = self.fusion_s(s_out)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        w_out = x
        #w_out = self.fusion_w(w_out)
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time


class Stage5_urban(nn.Module): # also named as stage5_urban_ad
    # mutual learning part
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        # 需要加载 stage3中训练得到的s_student的参数，然后加载stage2中的w_student 继续训练
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict) # 加载stage2中的w_student

        self.s_student = S_student(inputdim, outputdim)
        save_model_s = torch.load(model_config['stage3_path']) # we only need W_student part
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if k[:9]=='s_student' and k[10:] in model_dict_s.keys()}
        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s) # 加载stage2中的w_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        
    def forward(self,x,embedding):
        # the target is to train w_student
        s_out,s_decision_time = self.s_student.extract(x,embedding)
        #s_out = self.fusion_s(s_out)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        w_out = x
        #w_out = self.fusion_w(w_out)
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time


class Stage6(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage5_path']) # we only need W_student part
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict) # 加载stage2中的w_student
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        
        save_model_s = torch.load(model_config['stage3_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if k[:9]=='s_student' and k[10:] in model_dict_s.keys()}
        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s)
        # model_dict =  self.w_student.state_dict()
        
    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label

class Stage6_urban(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage5_path']) # we only need W_student part
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict) # 加载stage2中的w_student
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        
        save_model_s = torch.load(model_config['stage3_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if k[:9]=='s_student' and k[10:] in model_dict_s.keys()}
        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s)
        # model_dict =  self.w_student.state_dict()
        
    def forward(self, x, embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label

class SimilarityNet(nn.Module):
    def __init__(self, inputdim):
        super().__init__()
        self.sim = nn.Sequential(
            nn.Linear(inputdim*2, inputdim * 4),
            nn.BatchNorm1d(inputdim * 4),
            nn.ReLU(),
            nn.Linear(inputdim * 4, inputdim * 4),
            nn.BatchNorm1d(inputdim * 4),
            nn.ReLU(),
            nn.Linear(inputdim * 4, 2))
    def forward(self,x):
        x = self.sim(x)
        output = torch.softmax(x,dim=1)
        return output[:,0]

class Train_simnet(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.simNet = SimilarityNet(256)
        self.s_student = S_student(inputdim, outputdim)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        self.BN = nn.BatchNorm1d(256)
    def forward(self,x, embedding, label=None):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        x = x.transpose(1,2)
        x = self.BN(x)
        x = x.transpose(1,2)
        ans,ans_label = self.make_pair(x,label)
        sim_res = self.simNet(ans)
        return sim_res.squeeze(), ans_label
    
    def make_pair(self,x,label):
        batch,dim = label.shape
        # choose positive pair
        # print('label ',label)
        pair_ls_pos = None
        label_ls_pos = None
        pair_ls_neg = None
        label_ls_neg = None
        for i in range(batch):
            pos_index = label[i,:] > 0.5 # choose the positive frame
            pos_embedding = x[i,pos_index,:]
            if pos_embedding.shape[0] > 2: # if the positive frame is larger than 2
                if label_ls_pos == None:
                    pair_ls_pos = self.get_pos_pair(pos_embedding) # get positive pair
                    label_ls_pos = torch.ones(pair_ls_pos.shape[0]).cuda() # get positive label
                else:
                    tmp_ls = self.get_pos_pair(pos_embedding)
                    pair_ls_pos = torch.cat([pair_ls_pos,tmp_ls],dim=0)
                    label_ls_pos = torch.cat([label_ls_pos,torch.ones(tmp_ls.shape[0]).cuda()])
            neg_index = label[i,:] < 0.5 # choose negative frame
            neg_embedding = x[i,neg_index,:]
            if pos_embedding.shape[0]>1 and neg_embedding.shape[0]>1: # when positive and negative frame both exsit
                if label_ls_neg==None:
                    pair_ls_neg = self.get_neg_pair(pos_embedding,neg_embedding) # 
                    label_ls_neg = torch.zeros(pair_ls_neg.shape[0]).cuda()
                else:
                    tmp_ls = self.get_neg_pair(pos_embedding,neg_embedding)
                    pair_ls_neg = torch.cat([pair_ls_neg,tmp_ls],dim=0)
                    label_ls_neg = torch.cat([label_ls_neg,torch.zeros(tmp_ls.shape[0]).cuda()])
        # when positive frame is none or ,no negative frame, it will occur some mistake
        if pair_ls_neg!=None and pair_ls_pos!=None:
            if pair_ls_pos.shape[0]:
                pair_ls_pos = pair_ls_pos[torch.randperm(pair_ls_pos.shape[0]),:] # reshape the order
            if pair_ls_neg.shape[0]:
                pair_ls_neg = pair_ls_neg[torch.randperm(pair_ls_neg.shape[0]),:]
            min_batch = torch.min(torch.ones(1).cuda()*pair_ls_pos.shape[0],torch.ones(1).cuda()*pair_ls_neg.shape[0])
            min_batch = torch.min(torch.ones(1).cuda()*128,min_batch)
            ans = torch.cat([pair_ls_pos[:min_batch[0].long(),:],pair_ls_neg[:min_batch[0].long(),:]],dim=0)
            ans_label = torch.cat([label_ls_pos[:min_batch[0].long()],label_ls_neg[:min_batch[0].long()]],dim=0)
            per_ans = torch.randperm(ans.shape[0])
            ans = ans[per_ans,:]
            ans_label =ans_label[per_ans]
        else:
            if pair_ls_pos == None and pair_ls_neg == None:
                #print('x ',x.shape)
                # print('[x[0,0,:] ',x[0,0:2,:].shape)
                # print('x[1,124,:] ',x[1,-3:-1,:].shape)
                ans = torch.cat([x[0,0:2,:],x[0,-3:-1,:]],dim=1)
                # ans = ans.unsqueeze(0)
                ans_label = torch.zeros(2).cuda()
            elif pair_ls_neg == None:
                if pair_ls_pos.shape[0]:
                    pair_ls_pos = pair_ls_pos[torch.randperm(pair_ls_pos.shape[0]),:]
                #min_batch = torch.min(torch.ones(1).cuda()*64,torch.ones(1).cuda()*pair_ls_pos.shape[0])
                ans_label = label_ls_pos
                ans = pair_ls_pos
            else:
                # print('[x[0,0,:] ',x[0,0:2,:].shape)
                # print('x[1,124,:] ',x[1,-3:-1,:].shape)
                ans = torch.cat([x[0,0:2,:],x[0,-3:-1,:]],dim=1)
                # ans = ans.unsqueeze(0)
                ans_label = torch.zeros(2).cuda()
        # print('ans ',ans.shape) # 512
        # print('ans_label ',ans_label)
        # assert 1==2
        return ans,ans_label

    def get_pos_pair(self,pos):
        # print('pos ',pos.shape)
        index_ = torch.range(0,pos.shape[0]-1).long()
        # print('index_ ',index_)
        random_int = torch.randint(0,pos.shape[0]-1,(1,1))
        # print('random_int ',random_int)
        index_r = ((index_+random_int[0])%pos.shape[0]).long()
        # print('index_r ',index_r)
        pos1 = pos[index_]
        pos2 = pos[index_r]
        # print('pos1 ',pos1[0,:5])
        # print('pos2 ',pos2[0,:5])
        # assert 1==2
        ans = torch.cat([pos1,pos2],dim=1)
        return ans
    def get_neg_pair(self,pos,neg):
        # print('pos ',pos[0,:5])
        # print('neg ',neg[0,:5])
        # assert 1==2
        min_num = torch.min(torch.ones(1)*pos.shape[0], torch.ones(1)*neg.shape[0])
        # print(min_num)
        pos = pos[:min_num[0].long(),:]
        neg = neg[:min_num[0].long(),:]
        ans = torch.cat([pos,neg],dim=1)
        return ans


class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnMeanAndMax(),
            nn.Linear(nh, nout),
        )
        # print(self.net)

    def forward(self, x):
        # print('x_DiscConv ',x.shape)
        return self.net(x)

class S_student_ad(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.disconv = DiscConv(512,1)

    def forward(self, x, embedding): # 
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        E = x # use for discriminator
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0], decision_up, E

    def discriminator(self, E, is_source):
        d = self.disconv(E) # input the E to discriminator
        d = d.squeeze()
        D_src = F.mse_loss(d[is_source == 1.0], is_source[is_source == 1.0].float())
        D_tgt = F.mse_loss(d[is_source == 0], is_source[is_source == 0].float())
        if len(d[is_source == 1.0])==0:
            loss_D = D_tgt/2.0
        elif len(d[is_source == 0])==0:
            loss_D = D_src/2.0
        else:
            loss_D = (D_src.float() + D_tgt.float()) / 2.0
        return loss_D

    def extract(self, x, embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        kd_e = x
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        return kd_e, decision_time[:,:,0]

class Stage1_ad(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student_ad(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)
        self.disconv = DiscConv(512,1)
        self.s_student.load_state_dict(torch.load(model_config['s_student_path'])) # init s_student
        for p in self.s_student.parameters(): # fix the parameter of s_student
            p.requires_grad = False
        # self.fusion_s = nn.Linear(1024,256)
        # self.fusion_w = nn.Linear(1024,256)
    def forward(self,x,embedding):
        # the target is to train w_student
        s_out, s_decision_time = self.s_student.extract(x, embedding)
        #s_out = self.fusion_s(s_out)
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        E_ad = x # using to adverse
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        # print('x ',x.shape)
        # assert 1==2
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        w_out = x
        #w_out = self.fusion_w(w_out)
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time,E_ad
    
    def discriminator(self, E, is_source):
        d = self.disconv(E) # input the E to discriminator
        d = d.squeeze()
        D_src = F.mse_loss(d[is_source == 1.0], is_source[is_source == 1.0].float())
        D_tgt = F.mse_loss(d[is_source == 0], is_source[is_source == 0].float())
        if len(d[is_source == 1.0])==0:
            loss_D = D_tgt/2.0
        elif len(d[is_source == 0])==0:
            loss_D = D_src/2.0
        else:
            loss_D = (D_src.float() + D_tgt.float()) / 2.0
        return loss_D

class Stage2_urban(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.w_student = W_student(inputdim, outputdim)
        save_model = torch.load(model_config['stage1_path']) # we only need W_student part
        # print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        state_dict = {k[10:]:v for k,v in save_model.items() if k[:9]=='w_student' and k[10:] in model_dict.keys()}
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
    
    def forward(self,x,embedding):
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.w_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1) # 
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.w_student.gru.flatten_parameters()
        x, _ = self.w_student.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.w_student.fc(x)
        w_decision_time = torch.softmax(self.w_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        w_clip_decision = self.w_student.temp_pool(x, w_decision_time).clamp(1e-7, 1.).squeeze(1) # torch.Size([16, 2])
        w_decision_up = torch.nn.functional.interpolate(
                w_decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return w_decision_time,w_clip_decision,w_decision_up

class Stage3_ad(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)

        # load s_student
        save_model_s = torch.load(model_config['s_student_path']) # we only need W_student part
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if (k[:9]=='s_student') and (k[10:] in model_dict_s.keys())}

        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s) # init s_student_ad
        
        save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        #print('save_model ',save_model.keys())
        model_dict =  self.w_student.state_dict()
        # for key in model_dict.keys():
        #     print(model_dict[key][0])
        #     break
        state_dict = {k[10:]:v for k,v in save_model.items() if (k[:9]=='w_student') and (k[10:] in model_dict.keys())}
        #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        # print('state_dict ',state_dict.keys())
        # assert 1==2
        model_dict.update(state_dict)
        self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        
    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label

class Stage3_ad_from_w(nn.Module):
    def __init__(self,model_config, inputdim, outputdim, **kwargs):
        super().__init__()
        self.s_student = S_student(inputdim, outputdim)
        self.w_student = W_student(inputdim, outputdim)

        # load s_student
        save_model_s = torch.load(model_config['s_student_path']) # we only need W_student part
        model_dict_s =  self.s_student.state_dict()
        state_dict_s = {k[10:]:v for k,v in save_model_s.items() if (k[:9]=='s_student') and (k[10:] in model_dict_s.keys())}

        model_dict_s.update(state_dict_s)
        self.s_student.load_state_dict(model_dict_s) # init s_student_ad
        

        self.w_student.load_state_dict(torch.load(model_config['w_student_path'])) # init s_student
        # save_model = torch.load(model_config['stage2_path']) # we only need W_student part
        # #print('save_model ',save_model.keys())
        # model_dict =  self.w_student.state_dict()
        # # for key in model_dict.keys():
        # #     print(model_dict[key][0])
        # #     break
        # state_dict = {k[10:]:v for k,v in save_model.items() if (k[:9]=='w_student') and (k[10:] in model_dict.keys())}
        # #print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
        # # print('state_dict ',state_dict.keys())
        # # assert 1==2
        # model_dict.update(state_dict)
        # self.w_student.load_state_dict(model_dict)
        # model_dict =  self.w_student.state_dict()
        for p in self.w_student.parameters(): # fix the parameter of w_student
            p.requires_grad = False
        
    def forward(self,x,embedding):
        with torch.no_grad():
            pseudo_label,_ = self.w_student.extract(x,embedding) # get pseudo label
        batch, time, dim = x.shape
        x = x.unsqueeze(1) # (b,1,t,d) 
        x = self.s_student.features(x) # 
        x = x.transpose(1, 2).contiguous().flatten(-2) # 重新拷贝一份x,之后推平-2:-1之间的维度 # (b,125,512)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1, x.shape[1], 1)
        x = torch.cat((x, embedding), dim=2) # [B, T, 128 + emb_dim]
        if not hasattr(self, '_flattened'):
            self.s_student.gru.flatten_parameters()
        x, _ = self.s_student.gru(x) #  x  torch.Size([16, 125, 256])
        # x = self.gru(x) #  x  torch.Size([16, 125, 256])
        x = self.s_student.fc(x)
        decision_time = torch.softmax(self.s_student.outputlayer(x),dim=2) # x  torch.Size([16, 125, 2])
        decision_up = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2), # [16, 2, 125]
                time, # 501
                mode='linear',
                align_corners=False).transpose(1, 2) # 从125插值回 501 ?--> (16,501,2)
        return decision_time[:,:,0],decision_up,pseudo_label


def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)

def weight_init_(models):
        for m in models.modules():
            if isinstance(m,nn.Conv2d):
                print(1)
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight.data,1)
                nn.init.constant_(m.bias.data,0)

class MaskGenerator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.last_conv = nn.Sequential(nn.Conv2d(513, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                           nn.BatchNorm2d(256), nn.ReLU(),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                                           nn.BatchNorm2d(256), nn.ReLU(),
                                           nn.Conv2d(256, 2, kernel_size=1, stride=1),
                                           )
        self.creterion=nn.MultiLabelSoftMarginLoss()
        self.embed_fc = nn.Linear(128,125)
        weight_init_(self.last_conv)
    def forward(self, features, embedding, labels=None):
        # print('labels ',labels.shape)
        # print('labels value ',labels)
        # print('features ',features.shape)
        embedding = self.embed_fc(embedding)
        embedding = embedding.unsqueeze(1).unsqueeze(3)
        features = torch.cat([features,embedding],dim=1) # B,513,125,1
        x = self.last_conv(features)
        # print('x ',x.shape)
        # constant BG scores
        # bg = torch.ones_like(x[:, :1])
        # # print('bg ',bg.shape)
        # x = torch.cat([bg, x], 1) # the background class
        # print('x ',x.shape)
        bs, c, h, w = x.size() # b,1,h,w

        masks = F.softmax(x, dim=1) # 得到每一个像素点属于目标类或者背景类的概率

        # reshaping
        features = x.view(bs, c, -1) # 压缩h,w. 每个通道c代表一类
        masks_ = masks.view(bs, c, -1) # 压缩h,w. 每个通道c代表一类，里面每个元素代表其发生该类的概率
        # print('masks_ ',masks_[0])
        # print('masks_ ',masks_.shape)
        # calculate y_c
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1)) # 得到每一类的值
        # print('cls_1 ',cls_1[0])
        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), \
                            p=3, \
                            c=0.01)
        # print('cls_2 ',cls_2[0])
        # adding the losses together
        #cls = cls_1[:, 1:] + cls_2[:, 1:]
        cls = cls_1 + cls_2
        # print('cls ',cls[0])
        # foreground stats
        # masks_ = masks_[:, 1:]

        if labels is None:
            return None,masks
        # print('masks_ now ', masks_)
        # print('(masks_.mean(-1) * labels).sum(-1) ',(masks_.mean(-1) * labels).sum(-1))
        # print('labels.sum(-1) ',labels.sum(-1))
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)
        # print('cls_fg ',cls_fg)
        loss_cls=self.creterion(cls, labels)
        # print('loss_cls ',loss_cls)
        # assert 1==2
        mask_loss={'mask_loss':loss_cls.mean()+cls_fg.mean()} # +cls_fg.mean())*0.1
        return mask_loss,masks # 只返回目标声音概率的通道？
