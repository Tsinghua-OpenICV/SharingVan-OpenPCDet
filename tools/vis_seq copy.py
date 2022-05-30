import mayavi.mlab as mlab
import numpy as np
import torch
import os
import pickle as pkl
from visual_utils import visualize_utils as V
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

class KittiTrackingDataset:
    def __init__(self, root_path, pred_path):
        self.root_path = root_path
        self.preds = {}
        data = pkl.load(open(pred_path, 'rb'))
        for d in data:
            self.preds[d['frame_id']] = d
        frame_ids = self.preds.keys()
        frame_ids = list(frame_ids)
        self.frame_ids = sorted(frame_ids)
    def get_result(self,idx):
        points = self.get_lidar(idx)
        label = self.get_label(idx)
        pred = self.get_pred(idx)
        calib = self.get_calib(idx)
        locs_label = np.array([x.loc for x in label if x.cls_type != 'DontCare'])
        if locs_label.shape[0] == 0:
            return points, None, pred
        dims_label = np.array([[x.w, x.l, x.h, x.ry] for x in label if x.cls_type != 'DontCare'])
        locs_label_lidar = calib.rect_to_lidar(locs_label)
        label_box = np.concatenate((locs_label_lidar, dims_label), axis=1) 
        return points, label_box, pred
    def get_lidar(self, idx):
        lidar_file = os.path.join(self.root_path ,'velodyne' , ('%s.bin' % idx))
        assert os.path.exists(lidar_file)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    def get_label(self, idx):
        label_file = os.path.join(self.root_path ,'label_2_auxi' , ('%s.txt' % idx))
        assert os.path.exists(label_file)
        return object3d_kitti.get_objects_from_label(label_file)
    def get_calib(self, idx):
        calib_file = os.path.join(self.root_path , 'calib' , ('%s.txt' % idx))
        assert os.path.exists(calib_file)
        return calibration_kitti.Calibration(calib_file)
    def get_pred(self, idx):
        assert idx in self.preds
        return self.preds[idx]

root_path = '/root/OpenPCDet/data/kitti2/training'
pred_path = '/root/OpenPCDet/output-track/cfgs/kitti_models/pointrcnn_track_v5.1-backups/default/eval/eval_with_train/epoch_5/val/result.pkl'
pred_path2 = '/root/OpenPCDet/output/cfgs/kitti_models/pointrcnn0/default/eval/epoch_1/val/default/result.pkl'
output_path = '/root/OpenPCDet/vis_results'
output_path2 = '/root/OpenPCDet/vis_results2'
dataset = KittiTrackingDataset(root_path, pred_path)
dataset2 = KittiTrackingDataset(root_path, pred_path2)
name2int = {'Car':1, 'Pedestrian':2, 'Cyclist':3}
for frame_id in dataset.frame_ids:
    if int(frame_id) < 6700:
        continue
    points, label_box, pred = dataset.get_result(frame_id)
    points2, label_box2, pred2 = dataset2.get_result(frame_id)

    if pred['boxes_lidar'].shape == pred2['boxes_lidar'].shape:
        continue
    V.my_draw_scenes(
        points=points[:, :3], gt_boxes=label_box, ref_boxes=pred['boxes_lidar'],
        ref_scores=pred['score'], ref_labels=np.array([name2int[n] for n in pred['name']])
    )
    mlab.show(stop=True)
    output_dir = os.path.join(output_path, frame_id + '.png')
    mlab.savefig(output_dir)

    points2, label_box2, pred2 = dataset2.get_result(frame_id)

    V.my_draw_scenes(
        points=points2[:, :3], gt_boxes=label_box2, ref_boxes=pred2['boxes_lidar'],
        ref_scores=pred2['score'], ref_labels=np.array([name2int[n] for n in pred2['name']])
    )
    mlab.show(stop=True)
    output_dir2 = os.path.join(output_path2, frame_id + '.png')
    mlab.savefig(output_dir2)

    mlab.close(all=True)

