# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
# Import All Necessary Packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torchaudio
import torch
import torch.nn as nn
from sklearn import model_selection
from sklearn import metrics
from tabulate import tabulate # tabulate print
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
import torch.nn.functional as F
from pathlib import Path
import h5py
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
warnings.filterwarnings("ignore") # Ignore All Warnings
# df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
# df.head()

# wave = torchaudio.load("../input/urbansound8k/fold1/102106-3-0-0.wav")
# plt.plot(wave[0].t().numpy())
# print(wave[0].shape) # torch.Size([2, 72324]) 2 channels, 72324 sample_rate

# device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x
    return x.to(device)

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
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
        
        self.init_weights()
        
    def init_weights(self):
        
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
        else:
            raise Exception('Incorrect argument!')
        
        return x


class Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn10, self).__init__()

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

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
   
class Cnn14_emb128(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14_emb128, self).__init__()
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
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        # print('x.shape ',x.shape)
        # assert 1==2
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
def save_model(state, filename):
    torch.save(state, filename)
    print("-> Model Saved")

def get_embedding():
    file_write_obj1 = open("/apdcephfs/share_1316500/donchaoyang/code2/data/embeddings/spk_embed.128.txt", 'w')
    audio_ls = os.listdir('/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio4')
    path = []
    checkpoint = torch.load("/apdcephfs/share_1316500/donchaoyang/code2/data/pre_train_model/ft_local/CNN14_emb128_mAP=0.412.pth", map_location=torch.device("cpu"))
    model = Cnn14_emb128(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
    # checkpoint = torch.load("/apdcephfs/share_1316500/donchaoyang/code2/data/pre_train_model/ft_local/Cnn10_mAP=0.380.pth", map_location=torch.device("cpu"))
    # model = Cnn10(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
    #     fmax=14000, classes_num=527)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    for audio in audio_ls:
        ph = '/apdcephfs/share_1316500/donchaoyang/code2/data/reference/audio4/' + audio
        path.append(ph)
    for i in range(len(path)):
        print(path[i])
        (waveform, _) = librosa.core.load(path[i], sr=32000, mono=True)
        waveform = waveform[None, :]    # (1, audio_length)
        # print(waveform.shape)
        waveform = move_data_to_device(waveform, device)
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(waveform, None)
            #assert 1==2
            embedding = batch_output_dict['embedding'].detach().cpu().numpy()
            # print(embedding.shape)
            # assert 1==2
            basename = Path(path[i]).name
            file_write_obj1.write(basename+'\t')
            ft = ''
            for t in embedding[0,:]:
                ft = ft+str(t)+' '
            file_write_obj1.write(ft)
            file_write_obj1.write('\n')
    file_write_obj1.close()

def get_mel_mfcc():
    # h5_name = '/home/ydc/wsed/target_sound_detection/data/features/mel_path.h5'
    # print(h5_name)
    # hf = h5py.File(h5_name, 'w')
    # hf.create_dataset(
    #     name='filename', 
    #     shape=(0,), 
    #     maxshape=(None,),
    #     dtype='S80')
    # hf.create_dataset(
    #     name='file_path', 
    #     shape=(0,), 
    #     maxshape=(None,),
    #     dtype='S160')
    df = pd.read_csv("/home/ydc/wsed/target_sound_detection/data/UrbanSound8K/metadata/UrbanSound8K.csv")
    files = df["slice_file_name"].values.tolist()
    folder_fold = df["fold"].values
    label = df["classID"].values.tolist()
    path = [os.path.join(FILE_PATH + "fold" + str(folder) + "/" + file) for folder, file in zip(folder_fold, files)]
    n = 0
    for i in range(len(path)):
        waveform, sr = torchaudio.load(path[i])
        print('sr ',sr)
        assert 1==2
        print(i, path[i])
        audio_mono = torch.mean(waveform, dim=0, keepdim=True)
        tempData = torch.zeros([1, 160000])
        if audio_mono.numel() < 160000:
            tempData[:, :audio_mono.numel()] = audio_mono
        else:
            tempData = audio_mono[:, :160000]
        audio_mono=tempData
        output = audio_mono.numpy()
        basename = Path(path[i]).name
        # ans_ =str(basename,encoding="ascii")
        # mel_specgram = torchaudio.transforms.MelSpectrogram(sr)(audio_mono)
        # mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        # mfcc = torchaudio.transforms.MFCC(sample_rate=sr)(audio_mono)
        # mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # new_feat = torch.cat([mel_specgram, mfcc], axis=1)
        # new_feat = new_feat.permute(0, 2, 1).squeeze()
        hf['filename'].resize((n+1,))
        hf['filename'][n] = basename.encode()

        hf['file_path'].resize((n+1,))
        hf['file_path'][n] = path[i].encode()
        n += 1

    # file_write_obj1.close()
if __name__ == "__main__":
    get_embedding() # Run function