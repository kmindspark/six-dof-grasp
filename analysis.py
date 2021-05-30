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

def proj_axes_from_trans_rot(trans, rot_euler, render_size):
    axes = np.float32([[1,0,0],[0,1,0],[0,0,-1]])*0.2
    rot_mat = R.from_euler('xyz', rot_euler).as_matrix()
    axes = rot_mat@axes
    axes += trans
    axes_projected = []
    for axis in axes:
        axes_projected.append(project_3d_point(world_to_cam, Vector(axis), render_size))
    axes_projected = np.array(axes_projected)
    center_projected = project_3d_point(world_to_cam, Vector(trans), render_size)
    center_projected = tuple(center_projected)
    return center_projected, axes_projected

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

def run_inference(model, img, world_to_cam, gt_rot=None, output_dir='vis'):
    img_t = transform(img)
    img_t = img_t.cuda().unsqueeze(0)
    H,W,C = img.shape
    render_size = (W,H)
    heatmap, preds = model(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    pred_z_rot = preds.detach().cpu().numpy()[:, 0].squeeze()
    pred_y_rot = preds.detach().cpu().numpy()[:, 1].squeeze()
    angle = preds.detach().cpu().numpy()[:, 2].squeeze()
    heatmap = heatmap[0][0]
    pred_y, pred_x = np.unravel_index(heatmap.argmax(), heatmap.shape)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(img, 0.65, heatmap, 0.35, 0)
    heatmap = cv2.arrowedLine(heatmap, (100,100), (pred_x, pred_y), (0,0,0), 1)
    cv2.putText(heatmap,"Pred Offset",(pred_x,pred_y-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    trans = trans_gt = np.zeros(3)
    rot_euler = np.array([0,pred_y_rot,pred_z_rot])
    center_projected_pred, axes_projected_pred = proj_axes_from_trans_rot(trans_gt, rot_euler, render_size)
    vis_pred = draw_angle(img.copy(),center_projected_pred,axes_projected_pred, angle)
    cv2.putText(vis_pred,"Pred",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    if gt_rot is not None:
        center_projected_gt, axes_projected_gt = proj_axes_from_trans_rot(trans_gt, rot_euler_gt, render_size)
        vis_gt = draw(img.copy(),center_projected_gt,axes_projected_gt, intensity=100)
        cv2.putText(vis_gt,"Ground Truth",(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        result = np.hstack((vis_gt, vis_pred))
    else:
        result = vis_pred
    result = np.hstack((result, heatmap))
    return result

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model = SixDOFNet()
    model.load_state_dict(torch.load('/host/checkpoints/orthogonal_cables_one_linear/model_2_1_19.pth'))
    torch.cuda.set_device(0)
    model = model.cuda()
    model.eval()
    dataset_dir = '/host/datasets/dset_test'
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
        try:
            img = cv2.imread(os.path.join(image_dir, f))
            H,W,C = img.shape
            render_size = (W,H)
            #img = cv2.resize(img, (200,200))
            label = np.load(os.path.join(labels_dir, '%05d.npy'%i), allow_pickle=True)
	    trans = label.item().get("trans")
	    rot = label.item().get("rot")
            pixel = label.item().get("pixel")
            pixel, center_projected, axes_projected = get_center_axes(pixel, rot_euler, trans, render_size, world_to_cam)
            vis = img.copy()
            vis = draw(vis, center_projected, axes_projected)
            vis = cv2.resize(vis,(200,200))

            #Still under construction 
            success_pred = model(img, rot)

            annotated_filename = "%05d.jpg"%i
            if success_pred < 0.5:
                cv2.imwrite('%s/%s/%s'%(output_dir, 'fail', annotated_filename), vis)
            else:
                cv2.imwrite('%s/%s/%s'%(output_dir, 'success', annotated_filename), vis)
        except:
            pass
