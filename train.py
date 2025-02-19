# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import *
from tubevit.model import *

from torchvision import transforms

import time

from scipy import stats
from scipy.optimize import curve_fit

import logging
from torch.utils.tensorboard import SummaryWriter
from utils import performance_fit
from utils import L1RankLoss

import random
import pandas as pd

from torchvision.models import ViT_B_16_Weights

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='a',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

def save_model(config, model, old_save_name, epoch, performance):
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if os.path.exists(old_save_name):
        os.remove(old_save_name)
    save_model_name = os.path.join(config.ckpt_path,
                                   config.model_name + '_' + config.database + '_FR_v' + str(
                                       config.exp_version) + '_epoch_%d_SRCC_%f.pth' % (
                                       epoch + 1, performance))
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_model_name)
    else:
        torch.save(model.state_dict(), save_model_name)
    return save_model_name

def main(config):
    set_logging(config)
    logging.info(config)
    writer = SummaryWriter(os.path.join(config.log_path, config.log_file[:-4]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    video_shape = [3, config.clip_num*config.read_num, 224, 224]
    model = Tube_fps_cross_fuse(
            video_shape=video_shape,
            num_layers=6,
            context_dim=1
        )

    if config.load_name is not None:
        weights = torch.load(config.load_name)
        model.load_state_dict(weights)

    if config.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.conv_base_lr, weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.decay_interval, gamma=config.decay_ratio) ###0416
    if config.loss_type == 'MSE':
        criterion = nn.MSELoss().to(device)
        print('MSE LOSS')
    elif config.loss_type == 'L1RankLoss':
        criterion = L1RankLoss()
        print('L1Rank LOSS')

    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))
    logging.info('Trainable params: %.2f million' % (param_num / 1e6))

    videos_dir = config.videos_dir
    datainfo = config.datainfo
    test_videos_dir = config.test_videos_dir
    test_datainfo = config.test_datainfo

    transformations_train = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize([config.imgsize, config.imgsize]),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])
    transformations_test = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize([config.imgsize, config.imgsize]),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

    if config.database == 'ETRI':
        trainset = FrameDataset_etri(videos_dir, datainfo, transformations_train, is_train=True, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.features_dir)
        valset = FrameDataset_etri(videos_dir, datainfo, transformations_test, is_train=False, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.features_dir)
        testset = FrameDataset_etri(test_videos_dir, test_datainfo, transformations_test, is_train=False, is_test=True, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.test_features_dir)
    else:
        trainset = FrameDataset(videos_dir, datainfo, transformations_train, is_train=True, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.features_dir)
        valset = FrameDataset(videos_dir, datainfo, transformations_test, is_train=False, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.features_dir)
        testset = FrameDataset(test_videos_dir, test_datainfo, transformations_test, is_train=False, is_test=True, clip_num=config.clip_num, read_num=config.read_num, features_dir=config.test_features_dir)

    ## dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size,
                                               shuffle=True, num_workers=config.num_workers, collate_fn=my_collate)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=1,
                                              shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=config.num_workers, collate_fn=my_collate)

    best_criterion = -1  # SROCC min
    old_save_name = 'None'
    best_result = [-1,-1,-1,-1]
    best_test_result = [-1, -1, -1, -1]

    for epoch in range(config.epochs):
        model.train()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        print(epoch)
        train_label, train_output = [], []
        for i, (video_dis, dmos, _, fea, fea_len) in enumerate(train_loader):
            video_dis = video_dis.to(device)
            labels = dmos.to(device).float()
            fea = fea.to(device)
            fea_len = fea_len.to(device)
            outputs = model(video_dis, fea, fea_len)
            optimizer.zero_grad()
            loss = criterion(labels, outputs)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())
            loss.backward()

            optimizer.step()

            if (i + 1) % (config.print_samples // config.train_batch_size) == 0:
                avg_loss_epoch = sum(batch_losses_each_disp) / (config.print_samples // config.train_batch_size)
                batch_losses_each_disp = []

                print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' % (
                    epoch + 1, config.epochs, i + 1, len(trainset) // config.train_batch_size, avg_loss_epoch))

            train_label.extend(dmos.numpy().tolist())
            train_output.extend(outputs.cpu().detach().numpy().tolist())

        session_end_time = time.time()
        avg_loss = sum(batch_losses) / (len(trainset) // config.train_batch_size)

        print('TrainingCostTime: {:.4f}'.format(session_end_time - session_start_time))
        print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

        scheduler.step()
        lr = scheduler.get_last_lr()
        print('The current learning rate is {:.06f}'.format(lr[0]))

        with torch.no_grad():
            model.eval()
            label = np.zeros([len(valset)])
            y_output = np.zeros([len(valset)])
            session_start_time = time.time()
            for i, (video_dis, dmos, _, fea, fea_len) in enumerate(val_loader):
                video_dis = video_dis.to(device)
                fea = fea.to(device)
                fea_len = fea_len.to(device)
                outputs = model(video_dis, fea, fea_len)

                label[i] = dmos.item()
                y_output[i] = outputs.item()

            session_end_time = time.time()
            print('Test CostTime: {:.4f}'.format(session_end_time - session_start_time))

            val_SRCC, val_KRCC, val_PLCC, val_RMSE = performance_fit(label, y_output)
            print(
                'The result on the val databaset: SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                    val_SRCC, val_KRCC, val_PLCC, val_RMSE * 100))
        val_loss = criterion(torch.FloatTensor(label), torch.FloatTensor(y_output))
        print('Val loss:{:.4f}'.format(val_loss.item()))

        selected_performance = [val_SRCC, val_KRCC, val_PLCC, val_RMSE * 100]
        if selected_performance[0] > best_criterion:
            print("Update best model using best_criterion in epoch {}".format(epoch + 1))
            best_criterion = selected_performance[0]
            best_result = selected_performance
            old_save_name = save_model(config, model, old_save_name, epoch, selected_performance[0])
            data_best = [str(epoch)] + best_result + best_test_result
        print(
            'the best SRCC: {:.4f}, KRCC: {:.4f}, PLCC: {:.4f}, and RMSE: {:.4f}'.format(
                best_result[0], best_result[1], best_result[2], best_result[3]))

    print('Training completed.')
    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)
    np.save(
        os.path.join(config.results_path, config.model_name + '_' + config.database + '_v' + str(config.exp_version)),
        best_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--database', type=str, default='LIVEHFR')
    parser.add_argument('--model_name', type=str, default='ResNet18')

    parser.add_argument('--conv_base_lr', type=float, default=0.0001)
    parser.add_argument('--datainfo', type=str)
    parser.add_argument('--test_datainfo', type=str)
    parser.add_argument('--videos_dir', type=str)
    parser.add_argument('--test_videos_dir', type=str)
    parser.add_argument('--features_dir', type=str, default='None')
    parser.add_argument('--test_features_dir', type=str, default='None')
    parser.add_argument('--decay_ratio', type=float, default=0.8)
    parser.add_argument('--decay_interval', type=int, default=100)
    parser.add_argument('--results_path', type=str, default='./output/result')
    parser.add_argument('--exp_version', type=int, default=1)
    parser.add_argument('--print_samples', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)

    parser.add_argument('--log_path', type=str, default="./output/")
    parser.add_argument('--log_file', type=str, default="debug.txt")
    parser.add_argument('--load_name', type=str, default=None)

    parser.add_argument('--ckpt_path', type=str, default='./output/ckpts')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--imgsize', type=int, default=224)

    parser.add_argument('--loss_type', type=str, default='MSE')

    parser.add_argument('--clip_num', type=int, default=1)
    parser.add_argument('--read_num', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1234)

    config = parser.parse_args()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    main(config)
