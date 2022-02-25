"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as trn
from PIL import Image


def transform(image):
    """ Transform a image by cv2.
    """
    image = image.numpy()
    image = np.moveaxis(image, 0, -1)
    img_size = image.shape[0]
    # random crop
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.8:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image


class PlaceDataset(Dataset):
    def __init__(self, data_root, train_file,  names_file, limit_per_class = None):
        self.names_list = []
        if names_file is not None:
            names_file_buf = open(names_file)
            line = names_file_buf.readline().strip()
            while line:
                self.names_list.append(line)
                line = names_file_buf.readline().strip()
        name_count = {}
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path = line
            image_label = int(self.names_list.index(image_path.split('/')[1]))
            if image_label not in name_count:
                name_count[image_label] = 0
            if not limit_per_class or name_count[image_label] < 10:
                self.train_list.append((image_path, image_label))
                name_count[image_label] +=1
            line = train_file_buf.readline().strip()
        self.centre_crop = trn.Compose([
                trn.Resize((256,256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = Image.open(image_path).convert('RGB')#cv2.imread(image_path)
        image = self.centre_crop(image).unsqueeze(0)
        # image = torch.from_numpy(image.astype(np.float32))
        image = torch.squeeze(image, 0)
        return image, image_label



class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, crop_eye=False, names_file = None, name_as_label=False, allow_unknown=False):
        self.names_list = []
        if names_file is not None:
            names_file_buf = open(names_file)
            line = names_file_buf.readline().strip()
            while line:
                self.names_list.append(line)
                line = names_file_buf.readline().strip()
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            if name_as_label:
                image_path = line.split(' ')[0]
                image_name = image_path.split('/')[0]
                if image_name in self.names_list or allow_unknown:
                    image_label = self.names_list.index(image_name) if image_name in self.names_list else -1
                    self.train_list.append((image_path, int(image_label)))
            else:
                image_path, image_label = line.split(' ')[:2]
                self.train_list.append((image_path, int(image_label)))
            line = train_file_buf.readline().strip()
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        if self.crop_eye:
            image = image[:60, :]
        #image = cv2.resize(image, (128, 128)) #128 * 128
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))

        return image, image_label


class ImageDataset_SST(Dataset):
    def __init__(self, data_root, train_file, exclude_id_set):
        self.data_root = data_root
        label_set = set()
        # get id2image_path_list
        self.id2image_path_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, label = line.split(' ')
            label = int(label)
            if label in exclude_id_set:
                line = train_file_buf.readline().strip()
                continue
            label_set.add(label)
            if not label in self.id2image_path_list:
                self.id2image_path_list[label] = []
            self.id2image_path_list[label].append(image_path)
            line = train_file_buf.readline().strip()
        self.train_list = list(label_set)
        print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_list = self.id2image_path_list[cur_id]
        if len(cur_image_path_list) == 1:
            image_path1 = cur_image_path_list[0]
            image_path2 = cur_image_path_list[0]
        else:
            training_samples = random.sample(cur_image_path_list, 2)
            image_path1 = training_samples[0]
            image_path2 = training_samples[1]
        image_path1 = os.path.join(self.data_root, image_path1)
        image_path2 = os.path.join(self.data_root, image_path2)
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image1 = transform(image1)
        image2 = transform(image2)
        if random.random() > 0.5:
            return image2, image1, cur_id
        return image1, image2, cur_id
