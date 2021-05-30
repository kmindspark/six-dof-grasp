import torch
import cv2
import time
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
import pickle
import os
from datetime import datetime

transform = transforms.Compose([transforms.ToTensor()])

def normalize(x):
    return F.normalize(x, p=1)

def gauss_2d_batch(width, height, sigma, U, V, normalize_dist=False):
    U.unsqueeze_(1).unsqueeze_(2)
    V.unsqueeze_(1).unsqueeze_(2)
    X,Y = torch.meshgrid([torch.arange(0., width), torch.arange(0., height)])
    X,Y = torch.transpose(X, 0, 1).cuda(), torch.transpose(Y, 0, 1).cuda()
    G=torch.exp(-((X-U.float())**2+(Y-V.float())**2)/(2.0*sigma**2))
    if normalize_dist:
        return normalize(G).double()
    return G.double()

def vis_gauss(gauss):
    gauss = gauss.squeeze(0).cpu().numpy()
    output = cv2.normalize(gauss, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('test.png', output)


class PoseDataset(Dataset):
	def __init__(self, dataset_dir, transform, img_height=200, img_width=200, gauss_sigma=5):
		self.transform = transform
		self.imgs = []
		self.img_height = img_height
		self.img_width = img_width
		self.gauss_sigma = gauss_sigma
		self.rots = []
		#self.pixels = []
		self.dz = []
		self.results = []
		result_folder = os.path.join(dataset_dir, 'results')
		actions_folder = os.path.join(dataset_dir, 'annots')
		img_folder = os.path.join(dataset_dir, 'images')
		for i in range(len(os.listdir(img_folder))-1):
			self.imgs.append(os.path.join(img_folder, '%05d.jpg'%i))
			action = np.load(os.path.join(actions_folder, '%05d.npy'%i), allow_pickle=True)
			result = np.load(os.path.join(result_folder, '%05d.npy'%i), allow_pickle=True)
			result = result.astype(np.double)
			rot = action.item().get("rot")
			dz = action.item().get("angle") #change angle to be dz when actually using
			dz = dz.astype(np.float32)
			#pixel = (np.array([action.item().get("pixel")])*200/60).astype(int)
			#pixel[:,0] = np.clip(pixel[:, 0], 0, self.img_width-1)
			#pixel[:,1] = np.clip(pixel[:, 1], 0, self.img_height-1)
			rot = np.array([rot[2], rot[1]])
			rot = rot.astype(np.float32)
			#rot = np.array([rot[2], rot[1], d_rot[0]]) #add distractor cable rotation
			self.rots.append(torch.from_numpy(rot).cuda())
			#self.pixels.append(torch.from_numpy(pixel).cuda())
			self.dz.append(torch.from_numpy(dz).cuda())
			self.results.append(torch.from_numpy(result).cuda())

	def __getitem__(self, index):
		img_np = cv2.imread(self.imgs[index])
		img_np = cv2.resize(img_np, (200,200))
		img = self.transform(img_np)
		rot = self.rots[index]
		#pixel = self.pixels[index]
		dz = self.dz[index]
		result = self.results[index]
		#U = pixel[:,0]
		#V = pixel[:,1]
		#gaussian = gauss_2d_batch(self.img_width, self.img_height, self.gauss_sigma, U, V)

		return img, rot, dz, result #gaussian, rot, result

	def __len__(self):
		return len(self.rots)

if __name__ == '__main__':
        #TODO: need to test
	dset = PoseDataset('/host/datasets/cyl_white_kpt_test', transform)
	img, rot, label = dset[0]
