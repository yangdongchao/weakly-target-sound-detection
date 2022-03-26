# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang 
# @File    : train.py
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scipy
from h5py import File
from tqdm import tqdm
import torch.utils.data as tdata
import os
import h5py
import torchaudio
import random
import math
import pickle
event_labels = ['Alarm', 'Alarm_clock', 'Animal', 'Applause', 'Arrow', 'Artillery_fire', 
                'Babbling', 'Baby_laughter', 'Bark', 'Basketball_bounce', 'Battle_cry', 
                'Bell', 'Bird', 'Bleat', 'Bouncing', 'Breathing', 'Buzz', 'Camera', 
                'Cap_gun', 'Car', 'Car_alarm', 'Cat', 'Caw', 'Cheering', 'Child_singing', 
                'Choir', 'Chop', 'Chopping_(food)', 'Clapping', 'Clickety-clack', 'Clicking', 
                'Clip-clop', 'Cluck', 'Coin_(dropping)', 'Computer_keyboard', 'Conversation', 
                'Coo', 'Cough', 'Cowbell', 'Creak', 'Cricket', 'Croak', 'Crow', 'Crowd', 'DTMF', 
                'Dog', 'Door', 'Drill', 'Drip', 'Engine', 'Engine_starting', 'Explosion', 'Fart', 
                'Female_singing', 'Filing_(rasp)', 'Finger_snapping', 'Fire', 'Fire_alarm', 'Firecracker', 
                'Fireworks', 'Frog', 'Gasp', 'Gears', 'Giggle', 'Glass', 'Glass_shatter', 'Gobble', 'Groan', 
                'Growling', 'Hammer', 'Hands', 'Hiccup', 'Honk', 'Hoot', 'Howl', 'Human_sounds', 'Human_voice', 
                'Insect', 'Laughter', 'Liquid', 'Machine_gun', 'Male_singing', 'Mechanisms', 'Meow', 'Moo', 
                'Motorcycle', 'Mouse', 'Music', 'Oink', 'Owl', 'Pant', 'Pant_(dog)', 'Patter', 'Pig', 'Plop',
                'Pour', 'Power_tool', 'Purr', 'Quack', 'Radio', 'Rain_on_surface', 'Rapping', 'Rattle', 
                'Reversing_beeps', 'Ringtone', 'Roar', 'Run', 'Rustle', 'Scissors', 'Scrape', 'Scratch', 
                'Screaming', 'Sewing_machine', 'Shout', 'Shuffle', 'Shuffling_cards', 'Singing', 
                'Single-lens_reflex_camera', 'Siren', 'Skateboard', 'Sniff', 'Snoring', 'Speech', 
                'Speech_synthesizer', 'Spray', 'Squeak', 'Squeal', 'Steam', 'Stir', 'Surface_contact', 
                'Tap', 'Tap_dance', 'Telephone_bell_ringing', 'Television', 'Tick', 'Tick-tock', 'Tools', 
                'Train', 'Train_horn', 'Train_wheels_squealing', 'Truck', 'Turkey', 'Typewriter', 'Typing', 
                'Vehicle', 'Video_game_sound', 'Water', 'Whimper_(dog)', 'Whip', 'Whispering', 'Whistle', 
                'Whistling', 'Whoop', 'Wind', 'Writing', 'Yip', 'and_pans', 'bird_song', 'bleep', 'clink', 
                'cock-a-doodle-doo', 'crinkling', 'dove', 'dribble', 'eructation', 'faucet', 'flapping_wings', 
                'footsteps', 'gunfire', 'heartbeat', 'infant_cry', 'kid_speaking', 'man_speaking', 'mastication', 
                'mice', 'river', 'rooster', 'silverware', 'skidding', 'smack', 'sobbing', 'speedboat', 'splatter',
                'surf', 'thud', 'thwack', 'toot', 'truck_horn', 'tweet', 'vroom', 'waterfowl', 'woman_speaking']

event_to_id = {label : i for i, label in enumerate(event_labels)}
id_to_event = {i: label for i,label in enumerate(event_labels)}
event_to_time ={}
event_time_tsv = pd.read_csv('/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/train_time_total.tsv',sep='\t',usecols=[0,1])
labels = event_time_tsv['event_label']
times = event_time_tsv['time']
trans_labels = []
times_ls = []
for i in labels:
    trans_labels.append(i)
for i in times:
    times_ls.append(i)
for i in range(len(trans_labels)):
    if trans_labels[i] not in event_to_time.keys():
        event_to_time[trans_labels[i]] = times_ls[i]

event_labels_urban = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

event_to_id_urban = {label : i for i, label in enumerate(event_labels_urban)}
id_to_event_urban = {i: label for i,label in enumerate(event_labels_urban)}
def read_spk_emb_file_urban(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_id_dict = {}
    spk_emb_dict = {}
    with open(spk_emb_file_path, 'r') as file:
        for line in file:
            temp_line = line.strip().split('\t')
            file_id = os.path.basename(temp_line[0])
            emb = np.array(temp_line[1].split(' ')).astype(np.float)
            spk_id = int(file_id.split('-')[1])
            spk_id_label = id_to_event_urban[spk_id]
            spk_emb_dict[file_id] = emb
            if spk_id_label in spk_id_dict:
                spk_id_dict[spk_id_label].append(file_id)
            else:
                spk_id_dict[spk_id_label] = [file_id]
    return spk_emb_dict, spk_id_dict

def read_spk_emb_file(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_lab_dict = {}
    spk_emb_dict = {}
    with open(spk_emb_file_path, 'r') as file:
        for line in file:
            temp_line = line.strip().split('\t')
            file_id = os.path.basename(temp_line[0]) # get filename
            emb = np.array(temp_line[1].split(' ')).astype(np.float) # embedding
            spk_lab = file_id[:-5] # 
            # spk_id = event_to_id[spk_id]
            spk_emb_dict[file_id] = emb
            if spk_lab in spk_lab_dict:
                spk_lab_dict[spk_lab].append(file_id)
            else:
                spk_lab_dict[spk_lab] = [file_id]
        # print(len(spk_lab_dict.keys()))
        # assert 1==2
    return spk_emb_dict, spk_lab_dict

def read_spk_emb_file_by_h5(spk_emb_file_path):
    print('get spk_id_dict and spk_emb_dict')
    spk_id_dict = {}
    spk_emb_dict = {}
    mel_mfcc = h5py.File(spk_emb_file_path, 'r') # libver='latest', swmr=True
    file_name = np.array([filename.decode() for filename in mel_mfcc['filename'][:]])
    file_path = np.array([file_path.decode() for file_path in mel_mfcc['file_path'][:]])
    for i in range(file_name.shape[0]):
        file_id = file_name[i]
        emb = file_path[i]
        spk_id = int(file_id.split('-')[1])
        spk_id_label = id_to_event[spk_id]
        spk_emb_dict[file_id] = emb
        if spk_id_label in spk_id_dict:
            spk_id_dict[spk_id_label].append(file_id)
        else:
            spk_id_dict[spk_id_label] = [file_id]
    
    return spk_emb_dict,spk_id_dict
def time_to_frame(tim):
    return int(tim/0.08)  # 10/125

def label_smooth_for_neg(frame_level_label,alpha):
    frame_level_label = frame_level_label*alpha
    return frame_level_label


class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        #print(fname)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/' + fname[:-4]+'.txt'
        data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(125) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event

class HDF5Dataset_32000(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        self.is_w = True
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            if 'time' in hf.keys() and 'label' not in hf.keys():
                self.is_w = False # 给了强标签
                self.y = hf['time'][:].astype(np.float32)
            else:
                self.y = hf['label'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        #print(fname)
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        if self.is_w == False: # if give the strong label data
            tmp = self.y[index][0]
            # print(tmp)
            if tmp[0] == 0.0 and tmp[1]== 0.0:
                label = [0.0,1.0]
            else:
                label = [1.0,0.0]
            #assert 1==2
        else:
            label = self.y[index]
        data = torch.as_tensor(data).float()
        embedding = torch.as_tensor(embedding).float()
        label = torch.as_tensor(label).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, label, embedding, fname, target_event
        #return data, frame_level_label,slack_time, time, embedding, fname, target_event

class HDF5Dataset_32000_pkl(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000_pkl, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        self.is_w = True
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            if 'time' in hf.keys() and 'label' not in hf.keys():
                self.is_w = False # 给了强标签
                self.y = hf['time'][:].astype(np.float32)
            else:
                self.y = hf['label'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        #print(fname)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        if self.is_w == False: # if give the strong label data
            tmp = self.y[index][0]
            # print(tmp)
            if tmp[0] == 0.0 and tmp[1]== 0.0:
                label = [0.0,1.0]
            else:
                label = [1.0,0.0]
            #assert 1==2
        else:
            label = self.y[index]
        data = torch.as_tensor(data).float()
        embedding = torch.as_tensor(embedding).float()
        label = torch.as_tensor(label).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, label, embedding, fname, target_event


class HDF5Dataset_32000_urban(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000_urban, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file_urban(self._embedfile)
        self.is_w = True
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            if 'time' in hf.keys() and 'label' not in hf.keys():
                self.is_w = False # 给了强标签
                self.y = hf['time'][:].astype(np.float32)
            else:
                self.y = hf['label'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        #print(fname)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        if self.is_w == False: # if give the strong label data
            tmp = self.y[index][0]
            # print(tmp)
            if tmp[0] == 0.0 and tmp[1]== 0.0:
                label = [0.0,1.0]
            else:
                label = [1.0,0.0]
            #assert 1==2
        else:
            label = self.y[index]
        data = torch.as_tensor(data).float()
        embedding = torch.as_tensor(embedding).float()
        label = torch.as_tensor(label).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, label, embedding, fname, target_event

class HDF5Dataset_32000_urban_ad(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file_urban,
                 embedding_file_audioset,
                 transform=None):
        super(HDF5Dataset_32000_urban_ad, self).__init__()
        self._h5file = h5file
        self._embedfile_urban = embedding_file_urban
        self._embedfile_audioset = embedding_file_audioset
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict_urban, self.spk_lab_dict_urban = read_spk_emb_file_urban(self._embedfile_urban)
        self.spk_emb_dict_audioset, self.spk_lab_dict_audioset = read_spk_emb_file(self._embedfile_audioset)
        self.is_w = True
        with h5py.File(self._h5file, 'r') as hf:
            # self.X = hf['mel_feature'][:].astype(np.float32)
            if 'time' in hf.keys() and 'label' not in hf.keys():
                self.is_w = False # 给了强标签
                self.y = hf['time'][:].astype(np.float32)
            else:
                self.y = hf['label'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self.is_source = hf['is_source'][:].astype(np.float32)
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        is_source = self.is_source[index]
        if is_source > 0: # urban
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
            #mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        else: # audioset
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
        target_event = self.target_event[index]
        if is_source > 0: # urban
            embed_file_list = random.sample(self.spk_lab_dict_urban[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_urban[fl] / len(embed_file_list)
        else:
            embed_file_list = random.sample(self.spk_lab_dict_audioset[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_audioset[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        if self.is_w == False: # if give the strong label data
            tmp = self.y[index][0]
            # print(tmp)
            if tmp[0] == 0.0 and tmp[1]== 0.0:
                label = [0.0,1.0]
            else:
                label = [1.0,0.0]
            #assert 1==2
        else:
            label = self.y[index]
        data = torch.as_tensor(data).float()
        embedding = torch.as_tensor(embedding).float()
        label = torch.as_tensor(label).float()
        is_source = torch.as_tensor(is_source).float()
        # print('frame_level_time,frame_level_label',slack_time.shape,frame_level_label.shape)
        # print('frame_level_time,frame_level_label',slack_time.dtype,frame_level_label.dtype)
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, label, embedding, fname, target_event, is_source


class HDF5Dataset_32000_s(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000_s, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.ones(125) # 501 --> pooling 一次 到 250
        frame_level_label = label_smooth_for_neg(frame_level_label,0.1)
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 0.9

            r_start = random.randint(1,2)
            frame_level_label[min(start+r_start,124)] = 0.4 # noisy label
            frame_level_label[max(0,start-r_start)] = 0.6 # noisy label
            r_end = random.randint(1,2)
            frame_level_label[max(0,end-r_end)] = 0.4
            frame_level_label[min(end+r_end,124)] = 0.6

        for i in range(5):
            rr = random.randint(0,124)
            if frame_level_label[rr] > 0.5:
                frame_level_label[rr] = 0.4
            else:
                frame_level_label[rr] = 0.6
        clip_level_target = np.zeros(2)
        if frame_level_label.sum() > 0:
            clip_level_target[0]=1
        else:
            clip_level_target[1]=1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        clip_level_target = torch.as_tensor(clip_level_target).float()
        embedding = torch.as_tensor(embedding).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event, clip_level_target

class HDF5Dataset_32000_s_test(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000_s_test, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(125) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1.0
        clip_level_target = np.zeros(2)
        if frame_level_label.sum() > 0:
            clip_level_target[0]=1
        else:
            clip_level_target[1]=1
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        clip_level_target = torch.as_tensor(clip_level_target).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event, clip_level_target


class HDF5Dataset_32000_s_urban(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 mask_and_replace,
                 transform=None):
        super(HDF5Dataset_32000_s_urban, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.mask_and_replace = mask_and_replace
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file_urban(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.ones(125) # 501 --> pooling 一次 到 250
        frame_level_label = label_smooth_for_neg(frame_level_label,0.1)
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 0.9
            # r_start = random.randint(1,2)
            # frame_level_label[min(start+r_start,124)] = 0.4 # noisy label
            # frame_level_label[max(0,start-r_start)] = 0.6 # noisy label
            # r_end = random.randint(1,2)
            # frame_level_label[max(0,end-r_end)] = 0.4
            # frame_level_label[min(end+r_end,124)] = 0.6
        mask_num = 0
        rt_ls = [i for i in range(125)]
        # print('rt_ls ',rt_ls)
        if self.mask_and_replace == 0: # 不进行mask
            mask_num = 0
        elif self.mask_and_replace == 0.05: # 6
            mask_num = 6
        elif self.mask_and_replace == 0.1:
            mask_num = 12
        elif self.mask_and_replace == 0.15:
            mask_num = 18
        elif self.mask_and_replace == 0.2:
            mask_num = 24
        elif self.mask_and_replace == 0.25:
            mask_num = 30
        elif self.mask_and_replace == 0.35:
            mask_num = 42
        elif self.mask_and_replace == 0.45:
            mask_num = 54
        elif self.mask_and_replace == 0.55:
            mask_num = 80
        else:
            mask_num = 100

        # print('mask_num ',mask_num)
        mask_seq = random.sample(rt_ls, mask_num)
        # print('mask_seq ',mask_seq)
        # assert 1==2
        for i in range(len(mask_seq)):
            rr = mask_seq[i]
            if frame_level_label[rr] > 0.5:
                frame_level_label[rr] = 0.1
            else:
                frame_level_label[rr] = 0.9
        
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event

class HDF5Dataset_32000_s_urban_ad(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file_urban,
                 embedding_file_audioset,
                 transform=None):
        super(HDF5Dataset_32000_s_urban_ad, self).__init__()
        self._h5file = h5file
        self._embedfile_urban = embedding_file_urban
        self._embedfile_audioset = embedding_file_audioset
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict_urban, self.spk_lab_dict_urban = read_spk_emb_file_urban(self._embedfile_urban)
        self.spk_emb_dict_audioset, self.spk_lab_dict_audioset = read_spk_emb_file(self._embedfile_audioset)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self.is_source = hf['is_source'][:].astype(np.float32)
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        is_source = self.is_source[index]
        if is_source > 0: # urban
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
            #mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        else: # audioset
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
            #mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        #data = np.loadtxt(mel_path)
        time = self.y[index]
        target_event = self.target_event[index]
        if is_source > 0: # urban
            embed_file_list = random.sample(self.spk_lab_dict_urban[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_urban[fl] / len(embed_file_list)
        else:
            embed_file_list = random.sample(self.spk_lab_dict_audioset[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_audioset[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.ones(125) # 501 --> pooling 一次 到 250
        frame_level_label = label_smooth_for_neg(frame_level_label,0.1)
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 0.9

            r_start = random.randint(1,2)
            frame_level_label[min(start+r_start,124)] = 0.4 # noisy label
            frame_level_label[max(0,start-r_start)] = 0.6 # noisy label
            r_end = random.randint(1,2)
            frame_level_label[max(0,end-r_end)] = 0.4
            frame_level_label[min(end+r_end,124)] = 0.6

        for i in range(5):
            rr = random.randint(0,124)
            if frame_level_label[rr] > 0.5:
                frame_level_label[rr] = 0.4
            else:
                frame_level_label[rr] = 0.6
        
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        is_source = torch.as_tensor(is_source).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event, is_source

class HDF5Dataset_32000_s_test_urban_ad(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file_urban,
                 embedding_file_audioset,
                 transform=None):
        super(HDF5Dataset_32000_s_test_urban_ad, self).__init__()
        self._h5file = h5file
        self._embedfile_urban = embedding_file_urban
        self._embedfile_audioset = embedding_file_audioset
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict_urban, self.spk_lab_dict_urban = read_spk_emb_file_urban(self._embedfile_urban)
        self.spk_emb_dict_audioset, self.spk_lab_dict_audioset = read_spk_emb_file(self._embedfile_audioset)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self.is_source = hf['is_source'][:].astype(np.float32)
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        is_source = self.is_source[index]
        if is_source > 0: # urban
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
            #mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        else: # audioset
            mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
            f = open(mel_path,'rb')
            data = pickle.load(f)
            f.close()
        time = self.y[index]
        target_event = self.target_event[index]
        if is_source > 0: # urban
            embed_file_list = random.sample(self.spk_lab_dict_urban[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_urban[fl] / len(embed_file_list)
        else:
            embed_file_list = random.sample(self.spk_lab_dict_audioset[target_event], 1)
            embedding = np.zeros(128)
            for fl in embed_file_list:
                embedding = embedding + self.spk_emb_dict_audioset[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.ones(125) # 501 --> pooling 一次 到 250
        frame_level_label = label_smooth_for_neg(frame_level_label,0.1)
        #print(target_event, event_to_time[target_event],slack_time[0])
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1.0
        
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        is_source = torch.as_tensor(is_source).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event, is_source

class HDF5Dataset_32000_s_test_urban(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed (pretty likely)
    """
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_32000_s_test_urban, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_lab_dict = read_spk_emb_file_urban(self._embedfile)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['time'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/additional_urban_sed/feature_urban_pkl/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        time = self.y[index]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_lab_dict[target_event], 1)
        embedding = np.zeros(128)
        for fl in embed_file_list:
            embedding = embedding + self.spk_emb_dict[fl] / len(embed_file_list)
        # embedding = torch.as_tensor(embedding).float()
        frame_level_label = np.zeros(125) # 501 --> pooling 一次 到 250
        for i in range(60):
            if time[i,0] == -1:
                break
            if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
                break
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end] = 1.0
        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()
        embedding = torch.as_tensor(embedding).float()
        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, embedding, fname, target_event

class HDF5Dataset_strong_sed(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 transform=None):
        super(HDF5Dataset_strong_sed, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with h5py.File(self._h5file, 'r') as hf:
            self.X = hf['mel_feature'][:].astype(np.float32)
            self.y = hf['label'][:].astype(np.float32)
            self.events = np.array([target_event.decode() for target_event in hf['events'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        data = self.X[index]
        fname = self.filename[index]
        time = self.y[index]
        event = self.events[index]
        event_ls = event.split(',')
        if len(event_ls) == 0:
            event_ls.append(event) 
        # print(len(event_ls))
        # print(time.shape)
        # print(time[len(event_ls)])
        frame_level_label = np.zeros((125,len(event_labels))) # 501 --> pooling 2次 到 125
        for i in range(len(event_ls)):
            if event_ls[i] not in event_labels:
                continue
            class_id = event_to_id[event_ls[i]]
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end,class_id] = 1
        
        # print(time)
        # print(event)
        # assert time[len(event_ls),0] == -1 and time[len(event_ls),1] == -1

        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, fname

class HDF5Dataset_strong_sed_txt(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self,
                 h5file: File,
                 transform=None):
        super(HDF5Dataset_strong_sed_txt, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['label'][:].astype(np.float32)
            self.events = np.array([target_event.decode() for target_event in hf['events'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len

    def __getitem__(self, index): 
        fname = self.filename[index]
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        time = self.y[index]
        event = self.events[index]
        event_ls = event.split(',')
        if len(event_ls) == 0:
            event_ls.append(event) 
        # print(len(event_ls))
        # print(time.shape)
        # print(time[len(event_ls)])
        frame_level_label = np.zeros((125,len(event_labels))) # 501 --> pooling 2次 到 125
        for i in range(len(event_ls)):
            if event_ls[i] not in event_labels:
                continue
            class_id = event_to_id[event_ls[i]]
            start = time_to_frame(time[i,0])
            end = min(125,time_to_frame(time[i,1]))
            frame_level_label[start:end,class_id] = 1
        
        # print(time)
        # print(event)
        # assert time[len(event_ls),0] == -1 and time[len(event_ls),1] == -1

        data = torch.as_tensor(data).float()
        frame_level_label = torch.as_tensor(frame_level_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, frame_level_label, time, fname


class HDF5Dataset_join(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """ 
    def __init__(self,
                 h5file: File,
                 embedding_file,
                 transform=None):
        super(HDF5Dataset_join, self).__init__()
        self._h5file = h5file
        self._embedfile = embedding_file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        self.spk_emb_dict, self.spk_id_dict = read_spk_emb_file(self._embedfile) # 
        self.embed_mel_file = self._embedfile[:-4]  + '.h5'
        self.embeddings_mel = h5py.File(self.embed_mel_file,'r',libver='latest', swmr=True)
        #self.X = h5py.File('/apdcephfs/share_1316500/donchaoyang/code2/data/feature_32000/merge.h5','r',libver='latest', swmr=True)
        with h5py.File(self._h5file, 'r') as hf:
            self.y = hf['label'][:].astype(np.float32)
            self.target_event = np.array([target_event.decode() for target_event in hf['target_event'][:]])
            self.filename = np.array([filename.decode() for filename in hf['filename'][:]])
            self._len = self.filename.shape[0]
        
    def __len__(self):
        return self._len
    
    def __getitem__(self, index): 
        fname = self.filename[index]
        # mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_txt_32000/' + fname[:-4]+'.txt'
        # data = np.loadtxt(mel_path)
        mel_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/feature_pkl_32000/' + fname[:-4]+'.pkl'
        f = open(mel_path,'rb')
        data = pickle.load(f)
        f.close()
        #data = self.X[fname][()]
        target_event = self.target_event[index]
        embed_file_list = random.sample(self.spk_id_dict[target_event], 1)
        # embedding = np.loadtxt('/apdcephfs/share_1316500/donchaoyang/code2/data/embeddings/mel_txt/'+embed_file_list[0][:-4]+'.txt')
        embed_path = '/apdcephfs/share_1316500/donchaoyang/code2/data/embeddings/mel_pkl/'+embed_file_list[0][:-4]+'.pkl'
        f_e = open(embed_path,'rb')
        embedding = pickle.load(f_e)
        f_e.close()
        # embedding = self.spk_emb_dict[embed_file_list[0]]
        embed_label = np.zeros(10)
        label = self.y[index]
        # frame_level_label = np.zeros(125) # 501 --> pooling 一次 到 250
        # for i in range(60):
        #     if time[i,0] == -1:
        #         break
        #     if time[0,0]== 0.0 and time[0,1] == 0.0 and time[1,0] == -1:
        #         break
        #     start = time_to_frame(time[i,0])
        #     end = min(125,time_to_frame(time[i,1]))
        #     frame_level_label[start:end] = 1
        data = torch.as_tensor(data).float()
        embedding = torch.as_tensor(embedding).float()
        label = torch.as_tensor(label).float()
        embed_label = torch.as_tensor(embed_label).float()

        if self._transform:
            data = self._transform(data) # data augmentation
        return data, label, embedding,embed_label, fname, target_event


class MinimumOccupancySampler(tdata.Sampler):
    """
        docstring for MinimumOccupancySampler
        samples at least one instance from each class sequentially
    """
    def __init__(self, labels, sampling_mode='same', random_state=None):
        self.labels = labels
        data_samples, n_labels = labels.shape # get number of label ,and the dim of label
        label_to_idx_list, label_to_length = [], []
        self.random_state = np.random.RandomState(seed=random_state)
        for lb_idx in range(n_labels): # look for all class
            label_selection = labels[:, lb_idx] # select special class on all labels
            if scipy.sparse.issparse(label_selection):
                label_selection = label_selection.toarray()
            label_indexes = np.where(label_selection == 1)[0] # find all audio, where include the special class
            self.random_state.shuffle(label_indexes) # shuffle these index
            label_to_length.append(len(label_indexes))
            label_to_idx_list.append(label_indexes)

        self.longest_seq = max(label_to_length) # find the longest class
        self.data_source = np.zeros((self.longest_seq, len(label_to_length)),
                                    dtype=np.uint32) # build a matrix 
        # Each column represents one "single instance per class" data piece
        for ix, leng in enumerate(label_to_length):
            # Fill first only "real" samples
            self.data_source[:leng, ix] = label_to_idx_list[ix]

        self.label_to_idx_list = label_to_idx_list
        self.label_to_length = label_to_length

        if sampling_mode == 'same':
            self.data_length = data_samples
        elif sampling_mode == 'over':  # Sample all items
            self.data_length = np.prod(self.data_source.shape)

    def _reshuffle(self):
        # Reshuffle
        for ix, leng in enumerate(self.label_to_length):
            leftover = self.longest_seq - leng
            random_idxs = self.random_state.randint(leng, size=leftover)
            self.data_source[leng:,
                             ix] = self.label_to_idx_list[ix][random_idxs]

    def __iter__(self):
        # Before each epoch, reshuffle random indicies
        self._reshuffle()
        n_samples = len(self.data_source)
        random_indices = self.random_state.permutation(n_samples)
        data = np.concatenate(
            self.data_source[random_indices])[:self.data_length]
        return iter(data)

    def __len__(self):
        return self.data_length


def getdataloader(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_urban(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_urban(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_urban_ad(data_file, embedding_file_urban, embedding_file_audioset, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_urban_ad(data_file, embedding_file_urban, embedding_file_audioset, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s_test(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s_test(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s_urban(data_file,embedding_file,mask_and_replace, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s_urban(data_file, embedding_file, mask_and_replace, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s_urban_ad(data_file,embedding_file_urban,embedding_file_audioset, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s_urban_ad(data_file, embedding_file_urban, embedding_file_audioset, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s_test_urban(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s_test_urban(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_s_test_urban_ad(data_file,embedding_file_urban, embedding_file_audioset, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_32000_s_test_urban_ad(data_file, embedding_file_urban ,embedding_file_audioset, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_strong_sed(data_file, transform=None, **dataloader_kwargs): # aims at strong sed
    dset = HDF5Dataset_strong_sed_txt(data_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def getdataloader_join(data_file,embedding_file, transform=None, **dataloader_kwargs):
    dset = HDF5Dataset_join(data_file, embedding_file, transform=transform)
    return tdata.DataLoader(dset, collate_fn=sequential_collate, **dataloader_kwargs)

def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist, batch_first=batch_first, padding_value=padding_value)
    return padded_seq

def sequential_collate(batches):
    seqs = []
    for data_seq in zip(*batches):
        # print('data_seq[0] ',data_seq[0].shape)
        if isinstance(data_seq[0],
                      (torch.Tensor)):  # is tensor, then pad
            data_seq = pad(data_seq)
        elif type(data_seq[0]) is list or type(
                data_seq[0]) is tuple:  # is label or something, do not pad
            data_seq = torch.as_tensor(data_seq)
        seqs.append(data_seq)
    return seqs
