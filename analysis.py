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

def draw_distractor(img, source_px, imgpts):
    imgpts = imgpts.astype(int)
    img = cv2.arrowedLine(img, tuple(source_px), tuple(imgpts[0].ravel()), (100,0,0), 2)
    return img

def draw_angle(img, source_px, imgpts, angle):
    imgpts = imgpts.astype(int)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[0].ravel()), (255,0,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[1].ravel()), (0,255,0), 2)
    img = cv2.line(img, tuple(source_px), tuple(imgpts[2].ravel()), (0,0,255), 2)
    other_point = rotate_around_point(tuple(imgpts[0].ravel()), angle, tuple(source_px))
    img = cv2.line(img, tuple(source_px), other_point, (100,0,0), 2)
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

def run_inference(model, img, world_to_cam, gt_rot=None, output_dir='vis'):
    img_t = transform(img)
    img_t = img_t.cuda().unsqueeze(0)
    H,W,C = img.shape
    render_size = (W,H)
    heatmap, preds = model(img_t)
    heatmap = heatmap.detach().cpu().numpy()
    pred_z_rot = preds.detach().cpu().numpy()[:, 0].squeeze()
    pred_y_rot = preds.detach().cpu().numpy()[:, 1].squeeze()
    #pred_d_y_rot = preds.detach().cpu().numpy()[:, 2].squeeze()
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
    #rot_euler = pred
    center_projected_pred, axes_projected_pred = proj_axes_from_trans_rot(trans_gt, rot_euler, render_size)
    vis_pred = draw_angle(img.copy(),center_projected_pred,axes_projected_pred, angle)
    #d_rot_euler = np.array([0,0,pred_d_y_rot])
    #_, d_axes_projected_pred = proj_axes_from_trans_rot(trans_gt, d_rot_euler, render_size)
    #vis_pred = draw_distractor(vis_pred, center_projected_pred, d_axes_projected_pred)
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
    test_dir = '/host/datasets/crops'
    #test_dir = '/host/datasets/cyl_white_kpt_train/images'
    #test_dir = '/host/datasets/real_twocable_crops'
    #test_dir = '/host/datasets/cyl_dr_test/images'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for idx, f in enumerate(sorted(os.listdir(test_dir))):
        try:
            img = cv2.imread(os.path.join(test_dir, f))
            print(os.path.join(test_dir, f))
            img = cv2.resize(img, (200,200))
            gt_trans = np.zeros(3)
            gt_rot = None
            vis = run_inference(model, img, world_to_cam, gt_rot, output_dir)
            print("Annotating %06d"%idx)
            annotated_filename = "%05d.jpg"%idx
            cv2.imwrite('%s/%s'%(output_dir, annotated_filename), vis)
        except:
            pass
