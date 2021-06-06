import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import SixDOFNet
from src.dataset import PoseDataset, transform
import numpy as np
from mathutils import *
from scipy.spatial.transform import Rotation as R

def draw(img, source_px, imgpts, intensity=255):
    imgpts = imgpts.astype(int)
    source_px = tuple(source_px)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[0].ravel()), (intensity,0,0), 2)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[1].ravel()), (0,intensity,0), 2)
    img = cv2.arrowedLine(img, source_px, tuple(imgpts[2].ravel()), (0,0,intensity), 2)
    return img

def project_3d_point(transformation_matrix,p,render_size):
    p1 = transformation_matrix @ Vector((p.x, p.y, p.z, 1))
    p2 = Vector(((p1.x/p1.w, p1.y/p1.w)))
    p2 = (np.array(p2) - (-1))/(1 - (-1)) # Normalize -1,1 to 0,1 range
    pixel = [int(p2[0] * render_size[0]), int(render_size[1] - p2[1]*render_size[1])]
    return pixel

def get_center_axes(pixel, rot_euler, trans, render_size, world_to_cam):
    rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
    axes = np.float32([[1,0,0],[0,1,0],[0,0,-1]])*0.3
    axes = rot_mat@axes
    axes += trans
    axes_projected = []
    center_projected = project_3d_point(world_to_cam, Vector(trans), render_size)
    for axis in axes:
        axes_projected.append(project_3d_point(world_to_cam, Vector(axis), render_size))
    axes_projected = np.array(axes_projected)
    center_projected = pixel.astype(int)
    pixel = (200/60)*pixel
    pixel = tuple(pixel.astype(int))
    return pixel, center_projected, axes_projected

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = SixDOFNet()
    model.load_state_dict(torch.load('/host/checkpoints/dummy_grasp_success/model_2_1_16_0.6968537352414008.pth'))
    torch.cuda.set_device(0)
    model = model.cuda()
    model.eval()
    dataset_dir = '/host/datasets/dummy_grasp_success/dummy_test'
    image_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'annots')
    world_to_cam = Matrix(np.load('%s/cam_to_world.npy'%(labels_dir)))
    output_dir = 'vis'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists('%s/fail'%(output_dir)):
        os.mkdir('%s/fail'%(output_dir))
    if not os.path.exists('%s/success'%(output_dir)):
        os.mkdir('%s/success'%(output_dir))
    for i, f in enumerate(sorted(os.listdir(image_dir))):
        img = cv2.imread(os.path.join(image_dir, f))
        H,W,C = img.shape
        render_size = (W,H)
        img = cv2.resize(img, (200,200))
        label = np.load(os.path.join(labels_dir, '%05d.npy'%i), allow_pickle=True)
        trans = label.item().get("trans")
        rot = label.item().get("rot")
        pixel = label.item().get("pixel")
        dz = label.item().get("angle")
        pixel, center_projected, axes_projected = get_center_axes(pixel, rot, trans, render_size, world_to_cam)
        vis = draw(img.copy(), center_projected, axes_projected)
        vis = cv2.resize(vis,(200,200))
        img_t = transform(img)
        img_t = img_t.cuda().unsqueeze(0)
        rot = np.array([rot[2], rot[1]])
        rot = np.float32(rot)
        dz = np.float32(dz)
        rot = torch.tensor(rot).cuda()
        dz = torch.tensor(dz).cuda()
        rot = torch.unsqueeze(rot,0)
        dz = torch.unsqueeze(dz,0)

        success_pred = model(img_t, rot, dz)
        print('Evaluating image ' + str(i))
        annotated_filename = "%05d.jpg"%i
        if success_pred < 0.5:
            cv2.imwrite('%s/%s/%s'%(output_dir, 'fail', annotated_filename), vis)
        else:
            cv2.imwrite('%s/%s/%s'%(output_dir, 'success', annotated_filename), vis)
