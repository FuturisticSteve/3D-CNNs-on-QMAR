import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from spatial_transforms import *
from temporal_transforms import *
from dataset import get_training_set, get_test_set, get_flow_training_set, get_flow_test_set
from utils import *
from train import train_epoch
import test

if __name__ == '__main__':
    opt = parse_opts()

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)

    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()



    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            Scale((opt.sample_size,opt.sample_size)),
            #RandomHorizontalFlip(),
            # RandomRotate(),
            # RandomResize(),
            #crop_method,
            # MultiplyValues(),
            # Dropout(),
            # SaltImage(),
            # Gaussian_blur(),
            # SpatialElasticDisplacement(),
            ToTensor(opt.norm_value)
        ])
        transform_flow = Compose([
            ToTensor(opt.norm_value)
        ])
        #temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        #target_transform = ClassLabel()
        if opt.model == 'resnet' or opt.model == 'slowfastnet':
            training_data = get_training_set(opt, spatial_transform)
        if opt.model == 'flow':
            training_data = get_flow_training_set(opt, transform_flow)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr', 'correlation'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening


        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

    if opt.test:
        test_logger = Logger(
            os.path.join(opt.result_path, 'test.log'), ['epoch', 'correlation'])
        spatial_transform = Compose([
            Scale((opt.sample_size, opt.sample_size)),
            # MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
            ToTensor(opt.norm_value)
        ])
        if opt.model == 'resnet' or opt.model == 'slowfastnet':
            test_data = get_test_set(opt, spatial_transform)
        if opt.model == 'flow':
            test_data = get_flow_test_set(opt, transform_flow)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=10,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)


    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)

        if opt.test:
            correlation = test.test(test_loader, model, opt, i, test_logger)
            is_best = correlation>best_prec1
            best_prec1 = max(correlation, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, is_best, opt, i)







