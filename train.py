import pickle
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from src.model import SixDOFNet
from src.dataset import PoseDataset, transform
MSE = torch.nn.MSELoss()
bceLoss = nn.BCELoss()

def angle_loss(a,b):
    return MSE(torch.rad2deg(a), torch.rad2deg(b))

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def forward(sample_batched, model):
    img, gt_rot, gt_dz, gt_result = sample_batched
    img = Variable(img.cuda() if use_cuda else img)
    pred_result = model.forward(img, gt_rot, gt_dz).double() #predict success given image and action
    gt_result = gt_result.double()
    loss = F.binary_cross_entropy_with_logits(pred_result, gt_result)
    return loss

def fit(train_data, test_data, model, epochs, checkpoint_path = ''):
    for epoch in range(epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            loss = forward(sample_batched, model)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.item()), end='')
            print('\r', end='')
        print('train loss:', train_loss/i_batch)

        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.item()
        print('test kpt loss:', test_loss/i_batch)
        if epoch%2 == 0:
            torch.save(model.state_dict(), checkpoint_path + '/model_2_1_' + str(epoch) + '_' + str(test_loss/i_batch) + '.pth')

# dataset
workers=0
dataset_dir = 'dummy_grasp_success'
output_dir = 'checkpoints'
save_dir = os.path.join(output_dir, dataset_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


train_dataset = PoseDataset('/host/datasets/dummy_grasp_success/dummy_train', transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
test_dataset = PoseDataset('/host/datasets/dummy_grasp_success/dummy_test', transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
use_cuda = torch.cuda.is_available()

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(0)

# model
model = SixDOFNet().cuda()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1.0e-4)

print(epochs)
fit(train_data, test_data, model, epochs=epochs, checkpoint_path=save_dir)
