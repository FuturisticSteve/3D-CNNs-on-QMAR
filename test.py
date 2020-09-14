import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import scipy.stats

from utils import AverageMeter


# def calculate_video_results(output_buffer, video_id, test_results):
#     video_outputs = torch.stack(output_buffer)
#     average_scores = torch.mean(video_outputs, dim=0)
#     print(average_scores)
#     sorted_scores, pred = torch.topk(average_scores, k=1)
#
#     video_results = []
#     for i in range(sorted_scores.size(0)):
#         video_results.append({
#             'score': float(sorted_scores[i])
#         })
#
#     test_results['results'][video_id] = video_results


def test(data_loader, model, opt, epoch, test_logger=None):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    test_results = []
    ground_truth = []
    previous_video_id = ''
    previous_camera_id = ''
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = inputs.cuda(non_blocking=True)
            inputs = Variable(inputs)
        outputs = model(inputs)
        if not opt.no_softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and (targets[0][j] != previous_video_id or targets[1][j] != previous_camera_id):
                video_outputs = torch.stack(output_buffer)
                average_scores = torch.mean(video_outputs, dim=0)
                score, pred = torch.topk(average_scores, k=1)
                test_results.append(pred[0].item())
                ground_truth.append(targets[2][j-1].item())
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[0][j]
            previous_camera_id = targets[1][j]


        batch_time.update(time.time() - end_time)
        end_time = time.time()

    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    score, pred = torch.topk(average_scores, k=1)
    test_results.append(pred[0].item())
    ground_truth.append(targets[2][j].item())


    print("test results:")
    print(test_results)
    print("ground truth:")
    print(ground_truth)
    correlation = scipy.stats.spearmanr(test_results, ground_truth).correlation
    test_logger.log({'epoch': epoch,
                     'correlation': correlation
                     })
    print('[{}]\t'
          'Testing correlation: {correlation:4f}\t'.format(
            epoch,
            correlation = correlation))

    return correlation


