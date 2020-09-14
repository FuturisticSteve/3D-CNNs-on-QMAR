import torch
from torch import nn

import resnet
import slowfastnet


def generate_model(opt):
    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from resnet import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)

    if opt.model == 'slowfastnet':
        model = slowfastnet.resnet101(
            class_num=opt.n_classes)

    if opt.model == 'flow':
        model = resnet.resnet101(
            num_classes=opt.n_classes,
            shortcut_type=opt.resnet_shortcut,
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration,
            in_planes=2)



    if not opt.no_cuda:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)

        if opt.pretrain_path:
            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes))
                model.classifier = model.classifier.cuda()
            elif opt.model == 'squeezenet':
                model.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
                model.classifier = model.classifier.cuda()
            elif opt.model == 'flow':
                model.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
                model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)
                model.fc = model.fc.cuda()
            else:
                model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)
                model.fc = model.fc.cuda()


            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path, map_location=torch.device('cpu'))
            #assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])
            model.fc = nn.Linear(model.fc.in_features, opt.n_classes)
            if opt.model == 'flow':
                model.conv1 = nn.Conv3d(2, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            parameters = model.parameters()
            return model, parameters

        model = model.cuda()
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            assert opt.arch == pretrain['arch']
            model.load_state_dict(pretrain['state_dict'])

            if opt.model in  ['mobilenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2']:
                model.classifier = nn.Sequential(
                                nn.Dropout(0.9),
                                nn.Linear(model.classifier[1].in_features, opt.n_finetune_classes)
                                )
            elif opt.model == 'squeezenet':
                model.classifier = nn.Sequential(
                                nn.Dropout(p=0.5),
                                nn.Conv3d(model.classifier[1].in_channels, opt.n_finetune_classes, kernel_size=1),
                                nn.ReLU(inplace=True),
                                nn.AvgPool3d((1,4,4), stride=1))
            else:
                model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()



