"""Image demo script."""
import argparse
import os

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_one_box
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from skeleton_visualizer import SkeletonVisualizer

det_transform = T.Compose([T.ToTensor()])


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--img-dir',
                    help='image folder',
                    default='',
                    type=str)
parser.add_argument('--out-dir',
                    help='output folder',
                    default='',
                    type=str)
parser.add_argument('--viz',
                    help='enable visualization?',
                    default=False,
                    type=bool)
parser.add_argument('--use-smplx',
                    help='Use SMPL-X instead of SMPL',
                    default=False,
                    type=bool)
opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
CKPT = './pretrained_models/hybrik_hrnet48_w3dpw.pth'
if opt.use_smplx:
    cfg_file = 'configs/smplx/256x192_hrnet_smplx_kid.yaml'
    CKPT = './pretrained_models/hybrikx_hrnet.pth'

cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

det_model = fasterrcnn_resnet50_fpn(pretrained=True)

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

det_model.cuda(opt.gpu)
hybrik_model.cuda(opt.gpu)
det_model.eval()
hybrik_model.eval()

files = sorted(os.listdir(opt.img_dir))

smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))
if opt.use_smplx:
    smpl_faces = torch.from_numpy(hybrik_model.smplx_layer.faces.astype(np.int32))

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        # is an image
        if file[:4] == 'res_':
            continue
        
        # process file name
        img_path = os.path.join(opt.img_dir, file)
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        # Run Detection
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        tight_bbox = get_one_box(det_output)  # xyxy

        # Run HybrIK
        # bbox: [x1, y1, x2, y2]
        pose_input, bbox, img_center = transformation.test_transform(
            input_image, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]
        pose_output = hybrik_model(
            pose_input, flip_test=True,
            bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
            img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
        )

        # Get 2D keypoints in relative image coordinates w.r.t. the human bbox
        uv_jts = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]
        confidence = pose_output.scores.reshape(29, 1)

        # Get 3D keypoints in meters w.r.t. the human pelvis
        transl = pose_output.transl.detach()
        xyz_jts = pose_output.pred_xyz_jts_24_struct.reshape(24, 3).detach().cpu()

        # Visualize
        if opt.viz:           
            image = input_image.copy()
            focal = 1000.0
            bbox_xywh = xyxy2xywh(bbox)

            focal = focal / 256 * bbox_xywh[2]

            vertices = pose_output.pred_vertices.detach()

            verts_batch = vertices
            transl_batch = transl

            # Visualize 2D skeleton keypoints and the human mesh over the image
            color_batch = render_mesh(
                vertices=verts_batch, faces=smpl_faces,
                translation=transl_batch,
                focal_length=focal, height=image.shape[0], width=image.shape[1])
            
            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
            image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).cpu().numpy()

            color = image_vis_batch[0]
            valid_mask = valid_mask_batch[0].cpu().numpy()
            input_img = image
            alpha = 0.9
            image_vis = alpha * color[:, :, :3] * valid_mask + (
                1 - alpha) * input_img * valid_mask + (1 - valid_mask) * input_img

            image_vis = image_vis.astype(np.uint8)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

            x = uv_jts[:, 0] * bbox_xywh[2]
            y = uv_jts[:, 1] * bbox_xywh[3]
            kpts = torch.zeros_like(uv_jts)
            kpts[:, 0] = x + bbox_xywh[0]
            kpts[:, 1] = y + bbox_xywh[1]
            kpts = kpts.detach().cpu().numpy()       

            # Visualize 2D skeleton
            skeleton_vis = SkeletonVisualizer()

            image_vis = skeleton_vis.draw_skeleton_on_image(image_vis, kpts, thickness=2, radius=4)

            res_path = os.path.join(opt.out_dir, basename)
            cv2.imwrite(res_path, image_vis)

            # Visualize 3D skeleton (Hybrik -> XYZ in meters RELATIVE TO PELVIS!)
            kpts_3D_smpl = xyz_jts.detach().cpu().numpy()
            skeleton_vis.plot_3d_skeleton(kpts_3D_smpl, title=f'{basename} - 3D Skeleton', figtitle=basename)