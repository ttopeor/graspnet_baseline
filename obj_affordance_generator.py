import os
import sys
import argparse
import numpy as np
import open3d as o3d
import torch

from graspnet_baseline.models.graspnet import GraspNet, pred_decode
from graspnet_baseline.utils.collision_detector import ModelFreeCollisionDetector
from graspnet_baseline.utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup

from object_segments.detect_obj import ObjDetectorSOLOv2

COCO_80_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)


class ObjectAffordanceGenerator:

    def __init__(
        self,
        graspnet_checkpoint_path,
        solov2_config_path=None,
        solov2_ckpt_path=None,
        target_classes=None,
        num_point=20000,
        num_view=300,
        collision_thresh=0.01,
        voxel_size=0.01,
        visualize=True
    ):

        self.graspnet_checkpoint_path = graspnet_checkpoint_path
        self.solov2_config_path = solov2_config_path
        self.solov2_ckpt_path = solov2_ckpt_path

        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.visualize = visualize

        if target_classes is None:
            target_classes = []
        self.target_classes = [cls.lower() for cls in target_classes]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = self.get_net()

        if (self.solov2_config_path is not None) and (self.solov2_ckpt_path is not None):
            self.detector = ObjDetectorSOLOv2(
                config_file=self.solov2_config_path,
                checkpoint_file=self.solov2_ckpt_path,
                target_class=None,  
                score_thr=0.5
            )
        else:
            self.detector = None

        self.pipeline = None
        self.depth_scale = None
        self.align = None


    def get_net(self):

        net = GraspNet(
            input_feature_dim=0,
            num_view=self.num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False
        )
        net.to(self.device)

        if not os.path.exists(self.graspnet_checkpoint_path):
            raise FileNotFoundError(f"GraspNet checkpoint not found at {self.graspnet_checkpoint_path}")

        checkpoint = torch.load(self.graspnet_checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print(f"[INFO] Loaded GraspNet checkpoint {self.graspnet_checkpoint_path} (epoch: {epoch})")

        net.eval()
        return net


    def process_data(self, color_image, depth_image, workspace_mask, intrinsic, depth_scale):

        color = color_image.astype(np.float32) / 255.0
        depth = depth_image  # (H,W) uint16

        H, W = depth.shape
        cam_info = CameraInfo(
            width=W,
            height=H,
            fx=intrinsic[0, 0],
            fy=intrinsic[1, 1],
            cx=intrinsic[0, 2],
            cy=intrinsic[1, 2],
            scale=(1.0 / depth_scale)
        )
        cloud = create_point_cloud_from_depth_image(depth, cam_info, organized=True)

        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]   # [N, 3]
        color_masked = color[mask]   # [N, 3]

        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = cloud_masked[idxs, :]
        color_sampled = color_masked[idxs, :]

        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self.device)

        end_points = {
            'point_clouds': cloud_sampled,
            'cloud_colors': color_sampled 
        }
        return end_points, cloud_o3d


    def get_grasps(self, end_points):

        with torch.no_grad():
            end_points = self.net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()  # (num_grasp, 8)
        gg = GraspGroup(gg_array)
        return gg


    def collision_detection(self, gg, cloud_points):

        mfcdetector = ModelFreeCollisionDetector(cloud_points, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg_filtered = gg[~collision_mask]
        return gg_filtered


    def vis_grasps(self, gg, cloud_o3d, topK=50):

        gg.nms()
        gg.sort_by_score()
        gg_topk = gg[:topK]
        grippers = gg_topk.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud_o3d, *grippers])

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    REPO_DIR = os.path.join(BASE_DIR, '..','..')           
    MODEL_DIR = os.path.join(REPO_DIR, 'model')
    graspnet_ckpt_path = os.path.join(MODEL_DIR, 'checkpoint-rs.tar')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--graspnet_ckpt', type=str, default=graspnet_ckpt_path)
    parser.add_argument('--num_point', type=int, default=20000)
    parser.add_argument('--num_view',  type=int, default=300)
    parser.add_argument('--collision_thresh', type=float, default=0.01)
    parser.add_argument('--voxel_size', type=float, default=0.01)
    parser.add_argument('--visualize', action='store_true', default=True)
    args = parser.parse_args()

    generator = ObjectAffordanceGenerator(
        graspnet_checkpoint_path=args.graspnet_ckpt,
        num_point=args.num_point,
        num_view=args.num_view,
        collision_thresh=args.collision_thresh,
        voxel_size=args.voxel_size,
        visualize=args.visualize
    )

