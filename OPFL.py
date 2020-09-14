from torchvision import get_image_backend
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torch
import os
import math
import json
import numpy as np
import copy
import functools
from PIL import ImageFile
from pathlib import Path
import opts
ImageFile.LOAD_TRUNCATED_IMAGES = True




# def load_annotation_data(data_file_path):
#     return np.load(data_file_path)

def make_dataset(annotation_path, subset):
    data = np.load(annotation_path, allow_pickle=True)
    dataset = []
    label = []
    if subset == 'testing':
        for item in data:
            if len(item) > 0:
                for clip in item[1]:
                    dataset.append(clip)
                    video_id = Path(clip[0]).parent.parent.parent.name
                    camera_id = Path(clip[0]).parent.parent.name
                    label.append([video_id, camera_id, int(item[0])])
    else:
        for item in data:
            if len(item) > 0:
                for clip in item[1]:
                    dataset.append(clip)
                    label.append(int(item[0]))

    return dataset, label


def pil_loader(path):
    flow_frame = np.load(path, allow_pickle=True)
    return flow_frame


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(clip_paths, image_loader):
    video_clip = []
    for frame_path in clip_paths:
        pil_frame = pil_loader(frame_path)
        video_clip.append(pil_frame)
    return video_clip


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


class OPFL(Dataset):
    """
    Args:
        root (string): Root directory path.
        subset: Specify the traininf set or testing set
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 annotation_path,
                 subset,
                 transform_flow=None,
                 get_loader=get_default_video_loader):
        self.data, self.label = make_dataset(annotation_path, subset)

        self.transform_flow = transform_flow
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clip_paths = self.data[index]

        label = self.label[index]
        clip = self.loader(clip_paths)
        clip = [self.transform_flow(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, label

    def __len__(self):
        return len(self.data)