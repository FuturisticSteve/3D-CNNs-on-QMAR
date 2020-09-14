import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F

from opts import parse_opts
from model import generate_model
from spatial_transforms import *
from temporal_transforms import *
from dataset import get_training_set, get_test_set, get_flow_training_set, get_flow_test_set
from utils import *
from train import train_epoch
import test
import resnet
import time

if __name__ == '__main__':
    opt = parse_opts()

    RGB_state_dict = "I:\\results-101\\results-Parkinson-Sit-Stand-fold2\model_best.pth"
    flow_state_dict = "I:\Project\\results-flow-tvl1\\results-Parkinson-Sit-Stand-fold2\model_best.pth"



    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    print(opt)

    torch.manual_seed(opt.manual_seed)

    RGB_model = resnet.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    parameters = RGB_model.parameters
    pytorch_total_params = sum(p.numel() for p in RGB_model.parameters() if
                               p.requires_grad)
    print("Total number of RGBmodel trainable parameters: ", pytorch_total_params)
    print(RGB_model)

    flow_model = resnet.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            in_planes=2)
    parameters = flow_model.parameters
    pytorch_total_params = sum(p.numel() for p in flow_model.parameters() if
                               p.requires_grad)
    print("Total number of RGBmodel trainable parameters: ", pytorch_total_params)
    print(flow_model)

    spatial_transform = Compose([
        Scale((opt.sample_size, opt.sample_size)),
        # MultiScaleCornerCrop(opt.scales, opt.sample_size, crop_positions=['c']),
        ToTensor(opt.norm_value)
    ])
    transform_flow = Compose([
        ToTensor(opt.norm_value)
    ])

    RGB_test_data = get_test_set(opt, spatial_transform)
    RGB_test_loader = torch.utils.data.DataLoader(
        RGB_test_data,
        batch_size=10,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)
    flow_test_data = get_flow_test_set(opt, transform_flow)
    flow_test_loader = torch.utils.data.DataLoader(
        flow_test_data,
        batch_size=10,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True)

    print('loading RGB checkpoint {}'.format(opt.resume_path))
    checkpoint = torch.load(RGB_state_dict)
    best_prec1 = checkpoint['best_prec1']
    opt.begin_epoch = checkpoint['epoch']
    RGB_model = nn.DataParallel(RGB_model, device_ids=None)
    RGB_model.load_state_dict(checkpoint['state_dict'])


    RGB_model.cuda()

    print('loading flow checkpoint {}'.format(opt.resume_path_flow))
    checkpoint = torch.load(flow_state_dict)
    best_prec1 = checkpoint['best_prec1']
    opt.begin_epoch = checkpoint['epoch']
    flow_model.load_state_dict(checkpoint['state_dict'])
    flow_model = nn.DataParallel(flow_model, device_ids=None)


    flow_model.cuda()


    print('test')

    RGB_model.eval()
    flow_model.eval()

    output_buffer = []
    RGB_test_results = []
    ground_truth = []
    prediction_rgb = []
    previous_video_id = ''
    previous_camera_id = ''
    for i, (inputs, targets) in enumerate(RGB_test_loader):

        with torch.no_grad():
            inputs = inputs.cuda(non_blocking=True)
            inputs = Variable(inputs)
        outputs = RGB_model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and (targets[0][j] != previous_video_id or targets[1][j] != previous_camera_id):
                video_outputs = torch.stack(output_buffer)
                average_scores = torch.mean(video_outputs, dim=0)
                score_rgb, pred_rgb = torch.topk(average_scores, k=1)
                RGB_test_results.append(average_scores)
                prediction_rgb.append(pred_rgb.tolist()[0])
                ground_truth.append(targets[2][j - 1].item())
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[0][j]
            previous_camera_id = targets[1][j]


    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    score_rgb, pred_rgb = torch.topk(average_scores, k=1)
    RGB_test_results.append(average_scores)
    prediction_rgb.append(pred_rgb.tolist()[0])
    ground_truth.append(targets[2][j].item())

    correlation_rgb = scipy.stats.spearmanr(prediction_rgb, ground_truth).correlation

    print("RGB test results:")
    print(RGB_test_results)
    print("ground truth:")
    print(ground_truth)

    output_buffer = []
    flow_test_results = []
    prediction_flow = []
    previous_video_id = ''
    previous_camera_id = ''
    for i, (inputs, targets) in enumerate(flow_test_loader):

        with torch.no_grad():
            inputs = inputs.cuda(non_blocking=True)
            inputs = Variable(inputs)
        outputs = flow_model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and (targets[0][j] != previous_video_id or targets[1][j] != previous_camera_id):
                video_outputs = torch.stack(output_buffer)
                average_scores = torch.mean(video_outputs, dim=0)
                score_flow, pred_flow = torch.topk(average_scores, k=1)
                prediction_flow.append(pred_flow.tolist()[0])
                flow_test_results.append(average_scores)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[0][j]
            previous_camera_id = targets[1][j]

    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    flow_test_results.append(average_scores)
    score_flow, pred_flow = torch.topk(average_scores, k=1)
    prediction_flow.append(pred_flow.tolist()[0])


    correlation_flow = scipy.stats.spearmanr(prediction_flow, ground_truth).correlation


    print("flow test results:")
    print(flow_test_results)
    print("ground truth:")
    print(ground_truth)

    predictions = []
    for i in range(len(flow_test_results)):
        fusion_outputs = torch.stack((RGB_test_results[i], flow_test_results[i]), dim=0)
        average_scores = torch.mean(fusion_outputs, dim=0)
        score, pred = torch.topk(average_scores, k=1)
        predictions.append(pred.tolist()[0])


    correlation = scipy.stats.spearmanr(predictions, ground_truth).correlation

    print('RGB correlation: {correlation:4f}\t'.format(correlation=correlation_rgb))
    print('Flow correlation: {correlation:4f}\t'.format(correlation=correlation_flow))
    print('Testing correlation: {correlation:4f}\t'.format(correlation=correlation))








