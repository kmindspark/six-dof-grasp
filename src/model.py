import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
sys.path.insert(0, '/host/src')
from resnet_dilated import Resnet34_8s

class SixDOFNet(nn.Module):
        def __init__(self, img_height=200, img_width=200):
                super(SixDOFNet, self).__init__()
                self.img_height = img_height
                self.img_width = img_width
                self.resnet = Resnet34_8s(num_classes=1)
                self.fc1 = nn.Linear(40003, 128)
                self.fc2 = nn.Linear(128, 1)
                self.sigmoid = nn.Sigmoid()
        def forward(self, img, rot, dz):               
                output = self.resnet(img)
                output = torch.flatten(output, 1)
                output = torch.cat((output, rot), 1)
                output = torch.cat((output, dz), 1)
                output = self.fc1(output)
                output = F.relu(output)
                output = self.fc2(output)
                output = self.sigmoid(output)
                return output

if __name__ == '__main__':
	model = SixDOFNet().cuda()
	x = torch.rand((1,3,200,200)).cuda()
	rot = torch.rand((1,2)).cuda()
	dz = torch.rand((1,1)).cuda()
	out = model.forward(x, rot, dz)
	print(out)
