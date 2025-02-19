# -*- coding: utf-8
import os

import torch
from torch.utils import data
import cv2
import numpy as np
import json
import pandas as pd
from PIL import Image

class FrameDataset(data.Dataset):

    def __init__(self, data_dir, json_path, transform, is_train, clip_num=1, is_test=False, start_num=1, read_num=24, features_dir=None):
        super(FrameDataset, self).__init__()

        with open(json_path, 'r') as f:
            mos_file_content = json.loads(f.read())
            if is_train:
                self.frame_names_dis = mos_file_content['train']['dis']
                self.score = mos_file_content['train']['mos']
            elif not is_test:
                self.frame_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']
            else:
                self.frame_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']

        self.frames_dir = data_dir
        self.transform = transform
        self.length = len(self.score)
        self.clip_num = clip_num
        self.start_num = start_num
        self.read_num = read_num
        self.is_test = is_test

        self.features_dir = features_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        label = torch.FloatTensor(np.array(float(self.score[idx]))) / 100.0
        if self.is_test:
            class_name = self.frame_names_dis[idx].split('-')[1]
            video_dir = os.path.join(self.frames_dir, class_name.replace('fps', 'Hz'))
            filename = os.path.join(video_dir, '{}fps-360-1920x1080'.format(self.frame_names_dis[idx][:-2]))
            frame_rate = int(class_name.split('Hz')[0])
        else:
            class_name = self.frame_names_dis[idx].split('_')[0]
            video_dir = os.path.join(self.frames_dir, class_name)
            filename = os.path.join(video_dir, '{}'.format(self.frame_names_dis[idx]))
            frame_rate = int(self.frame_names_dis[idx].split('_')[-1].split('fps')[0])

        sn = self.start_num   # start_num
        rn = self.read_num  # read_num
        frame_index = list(np.linspace(sn, frame_rate, num=rn, endpoint=False, dtype=int))

        frames = []
        for index in frame_index:
            framename = os.path.join(filename, '{:>05d}.jpg'.format(index))
            read_frame = self._read_frame(framename)
            frames.append(read_frame)

        frames = torch.cat(frames, 1)
        assert frames.shape[1] == rn

        if self.is_test:
            class_name = self.frame_names_dis[idx].split('-')[1]
            features_dir = self.features_dir
            videoname = os.path.join(features_dir, '{}fps-360-1920x1080'.format(self.frame_names_dis[idx][:-2]))
        else:
            class_name = self.frame_names_dis[idx].split('_')[0]
            features_dir = os.path.join(self.features_dir, class_name)
            videoname = os.path.join(features_dir, '{}'.format(self.frame_names_dis[idx]))
        features_dir = os.path.join(features_dir, videoname)
        fea = torch.from_numpy(np.load(features_dir + '/feature_resnet50_std.npy').astype(np.float32))  ### (seq, 2048)
        fea_len = fea.shape[0]
        return frames, label, self.frame_names_dis[idx], fea, fea_len

    def _read_frame(self, framename):
        cap_frame = cv2.imread(framename)
        cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)
        transformed_frame = self.transform(cap_frame)
        transformed_frame = torch.unsqueeze(transformed_frame, 1)
        return transformed_frame  ### CTHW
    
class FrameDataset_etri(data.Dataset):
    def __init__(self, data_dir, json_path, transform, is_train, clip_num=1, is_test=False, start_num=1, read_num=24, features_dir=None):
        super(FrameDataset_etri, self).__init__()

        with open(json_path, 'r') as f:
            mos_file_content = json.loads(f.read())
            if is_train:
                self.frame_names_dis = mos_file_content['train']['dis']
                self.score = mos_file_content['train']['mos']
            elif not is_test:
                self.frame_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']
            else:
                self.frame_names_dis = mos_file_content['test']['dis']
                self.score = mos_file_content['test']['mos']

        self.frames_dir = data_dir
        self.transform = transform
        self.length = len(self.score)
        self.clip_num = clip_num
        self.start_num = start_num
        self.read_num = read_num
        self.is_test = is_test

        self.features_dir = features_dir

        ori_name = ['Beauty_2160p_120hz', 'Dancers_2160p_60hz', 'Dinner_2160p_60hz', 'Discuss_2160p_60hz', 'Football_2160p_60hz',
            'HoneyBee_2160p_120hz', 'Jockey_2160p_120hz', 'Monkeys_2160p_60hz', 'Narrator_2160p_60hz','Ready_2160p_120hz',
             'Ritual_2160p_60hz', 'SeaRock_2160p_60hz', 'Toddler_2160p_60hz', 'Tunnel_2160p_60hz', 'Yacht_2160p_120hz']
        self.ori_framerate = [ori.split('_')[-1].split('hz')[0] for ori in ori_name]
        self.ori_class = [ori.split('_')[0] for ori in ori_name]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        label = torch.FloatTensor(np.array(float(self.score[idx]))) / 100.0
        if self.is_test:
            class_name = self.frame_names_dis[idx].split('-')[1]
            video_dir = os.path.join(self.frames_dir, class_name.replace('fps', 'Hz'))
            filename = os.path.join(video_dir, '{}fps-360-1920x1080'.format(self.frame_names_dis[idx][:-2]))
            frame_rate = int(class_name.split('Hz')[0])
        else:
            class_name = self.frame_names_dis[idx].split('_')[0]
            video_dir = os.path.join(self.frames_dir, class_name)
            filename = os.path.join(video_dir, '{}'.format(self.frame_names_dis[idx]))
            class_idx = self.ori_class.index(class_name)
            ori_rate = int(self.ori_framerate[class_idx])
            frame_rate = ori_rate

        sn = self.start_num   # start_num
        rn = self.read_num  # read_num
        frame_index = list(np.linspace(sn, frame_rate, num=rn, endpoint=False, dtype=int))

        frames = []
        for index in frame_index:
            if self.is_test:
                framename = os.path.join(filename, '{:>05d}.jpg'.format(index))
            else:
                framename = os.path.join(filename, '{:>05d}.png'.format(index))
            if not os.path.exists(framename):
                print(framename)
            read_frame = self._read_frame(framename)
            frames.append(read_frame)

        frames = torch.cat(frames, 1)
        assert frames.shape[1] == rn

        if self.is_test:
            class_name = self.frame_names_dis[idx].split('-')[1]
            features_dir = self.features_dir
            videoname = os.path.join(features_dir, '{}fps-360-1920x1080'.format(self.frame_names_dis[idx][:-2]))
        else:
            features_dir = self.features_dir
            videoname = os.path.join(features_dir, '{}'.format(self.frame_names_dis[idx]))
        features_dir = os.path.join(features_dir, videoname)
        fea = torch.from_numpy(np.load(features_dir + '/feature_resnet50_std.npy').astype(np.float32))  ### (seq, 2048)
        fea_len = fea.shape[0]
        return frames, label, self.frame_names_dis[idx], fea, fea_len

    def _read_frame(self, framename):
        cap_frame = cv2.imread(framename)
        cap_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGB)
        transformed_frame = self.transform(cap_frame)
        transformed_frame = torch.unsqueeze(transformed_frame, 1)
        return transformed_frame  ### CTHW



from torch.nn.utils.rnn import pad_sequence
def my_collate(data):
    frames = [item[0] for item in data]
    label = [item[1] for item in data]
    name = [item[2] for item in data]
    feature_s = [item[3] for item in data]
    fea_len = [item[4] for item in data]
    # print([item[0].shape for item in data])
    # print([item[3].shape for item in data])
    frames = pad_sequence(frames, batch_first=True)
    label = torch.tensor(label)
    feature_s = pad_sequence(feature_s, batch_first=True)
    fea_len = torch.tensor(fea_len)
    # print(frames.shape, feature_s.shape)
    return [frames, label, name, feature_s, fea_len]
