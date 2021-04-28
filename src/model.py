import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
sys.path.insert(0, '/host/src')
from resnet_dilated_multi import Resnet34_8s
#from resnet_dilated import Resnet34_8s

class SixDOFNet(nn.Module):
	def __init__(self, img_height=200, img_width=200):
			super(SixDOFNet, self).__init__()
			self.img_height = img_height
			self.img_width = img_width
			self.resnet = Resnet34_8s(num_classes=1)
			self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
			heatmap, segmap, cls = self.resnet(x)
			heatmaps = self.sigmoid(heatmap[:,:1, :, :])
			segmaps = self.sigmoid(segmap[:, :1, :, :])
			return heatmaps, segmaps, cls

#class SixDOFNet(nn.Module):
#	def __init__(self, num_keypoints=1, img_height=480, img_width=640):
#		super(SixDOFNet, self).__init__()
#		self.num_keypoints = num_keypoints
#		self.num_outputs = self.num_keypoints
#		self.img_height = img_height
#		self.img_width = img_width
#		self.resnet = Resnet34_8s(num_classes=1)
#		self.sigmoid = torch.nn.Sigmoid()
#	def forward(self, x):
#		output = self.resnet(x)
#		heatmaps = self.sigmoid(output[:,:self.num_keypoints, :, :])
#		return heatmaps
#

if __name__ == '__main__':
	model = SixDOFNet().cuda()
	x = torch.rand((1,3,200,200)).cuda()
	heatmap = model.forward(x)
	#print(model)
	#print(heatmap.shape)
	heatmap, cls = model.forward(x)
	print(heatmap.shape, cls.shape)
