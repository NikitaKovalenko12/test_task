# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:26:48 2020

@author: Nikita
"""
import os
import sys
import json
import torch
import numpy as np
import natsort
import torchvision
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset
import insightface

mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
    
valtest_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
dataset = CustomDataSet(sys.argv[1], transform = valtest_transforms)
batch_size = 256

test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0)



model = insightface.iresnet34(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2,bias=True)
model.features =  torch.nn.BatchNorm1d(2, eps=2e-05, momentum=0.9, affine=True, track_running_stats=True)
model.load_state_dict(torch.load('internship_model'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


test_predictions = []
for inputs in tqdm(test_dataloader):
    inputs = inputs.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
    
test_predictions = np.concatenate(test_predictions)
test_predictions = [ 'male' if i>0.5 else 'female' for i in test_predictions]

dir_list = os.listdir(sys.argv[1])

with open("process_results.json", "w") as write_file:
    json.dump(dict(zip(dir_list, test_predictions)), write_file)
