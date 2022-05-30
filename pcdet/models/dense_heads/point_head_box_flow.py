import torch

from ...utils import box_coder_utils, box_utils
from .point_head_template import PointHeadTemplate
from .flownet3d_kitti import Kitti_FlowNet3D
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import group_gather_by_index

class PointHeadBox(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.predict_boxes_when_training = predict_boxes_when_training
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )
        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.box_coder.code_size
        )
        self.flownet = Kitti_FlowNet3D().cuda()
        self.flownet.load_state_dict(torch.load("/root/OpenPCDet/flownet/kitti_models_rm_ground_v2/net_epe_0.1108.pth"))
        self.flownet.eval()

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True
        )

        return targets_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        point_loss_box, tb_dict_2 = self.get_box_layer_loss()

        point_loss = point_loss_cls + point_loss_box
        tb_dict.update(tb_dict_1)
        tb_dict.update(tb_dict_2)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
            point_features_auxi = batch_dict['point_features_auxi']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        point_cls_preds_auxi = self.cls_layers(point_features_auxi)  # (total_points, num_class)
        point_cls_preds_max_auxi, _ = point_cls_preds_auxi.max(dim=-1)
        batch_dict['point_cls_scores_auxi'] = torch.sigmoid(point_cls_preds_max_auxi)

        '''select out foreground point'''
        num_points = 4096
        batch_size = batch_dict['batch_size']
        cls = batch_dict['point_cls_scores'] > 0.5
        cls = cls.view(batch_size,-1)
        coords = batch_dict['point_coords'][:,1:]
        coords = coords.view(batch_size,-1,3)
        cls_auxi = batch_dict['point_cls_scores_auxi'] > 0.5
        cls_auxi = cls_auxi.view(batch_size, -1)
        coords_auxi = batch_dict['point_coords_auxi'][:, 1:]
        coords_auxi = coords_auxi.view(batch_size, -1, 3)
        for k in range(0,batch_size):
            coords_i = coords[k,:,:]
            low_num = 0 # deal with fg_points < num_points
            cls_i = cls[k,:]
            fg_coords = coords_i[cls_i]
            if fg_coords.shape[0] < num_points:
                low_num = fg_coords.shape[0]
                if low_num < 100:
                    continue
                while(fg_coords.shape[0] < num_points):
                    fg_coords = torch.cat((fg_coords,fg_coords[0:min(fg_coords.shape[0],num_points-fg_coords.shape[0])]),0)

            part1 = torch.arange(0,num_points)
            part2 = torch.arange(fg_coords.shape[0]-num_points,fg_coords.shape[0])
            coords1 = fg_coords[part1,:]
            coords2 = fg_coords[part2,:]
            features1 = torch.zeros(1, 3, num_points)
            features2 = torch.zeros(1, 3, num_points)

            coords_i_auxi = coords_auxi[k, :]
            low_num_auxi = 0  # deal with fg_points < num_points
            cls_i_auxi = cls_auxi[k,:]
            fg_coords_auxi = coords_i_auxi[cls_i_auxi]
            if fg_coords_auxi.shape[0] < num_points:
                low_num_auxi = fg_coords_auxi.shape[0]
                if low_num_auxi <100:
                    continue
                while fg_coords_auxi.shape[0] < num_points:
                    fg_coords_auxi = torch.cat((fg_coords_auxi, fg_coords_auxi[0:min(fg_coords_auxi.shape[0],num_points - fg_coords_auxi.shape[0])]), 0)
            part1_auxi = torch.arange(0, num_points)
            part2_auxi = torch.arange(fg_coords_auxi.shape[0] - num_points, fg_coords_auxi.shape[0])
            coords1_auxi = fg_coords_auxi[part1_auxi, :]
            coords2_auxi = fg_coords_auxi[part2_auxi, :]
            features1_auxi = torch.zeros(1,3, num_points)
            features2_auxi = torch.zeros(1,3, num_points)


            pred_flow1 = self.flownet(coords1_auxi.view(1,-1,3).permute(0,2,1).contiguous().cuda(), coords1.view(1,-1,3).permute(0,2,1).contiguous().cuda(), features1_auxi.cuda(), features1.cuda())
            pred_flow2 = self.flownet(coords2_auxi.view(1,-1,3).permute(0,2,1).contiguous().cuda(), coords2.view(1,-1,3).permute(0,2,1).contiguous().cuda(), features2_auxi.cuda(), features2.cuda())
            pred_flow1 = pred_flow1.view(3, -1).permute(1,0)
            pred_flow2 = pred_flow2.view(3, -1).permute(1,0)

            pred_flow = torch.zeros(fg_coords_auxi.shape).cuda()
            pred_flow[part1_auxi,:] = pred_flow1
            pred_flow[part2_auxi, :] = pred_flow2
            if low_num_auxi:
                pred_flow = pred_flow[0:low_num_auxi,:]
                fg_coords_auxi = fg_coords_auxi[0:low_num_auxi,:]
            fg_coords_auxi = fg_coords_auxi + pred_flow
            coords_i_auxi[cls_i_auxi,:] = fg_coords_auxi
            coords_auxi[k,:,:] = coords_i_auxi

        batch_dict['point_coords_auxi'][:,1:] = coords_auxi.view(-1, 3)
        dist, ind = three_nn(coords.contiguous(), coords_auxi.contiguous())
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        new_features = torch.sum(group_gather_by_index(point_features_auxi.view(batch_size,-1,point_features_auxi.shape[1]).permute(0,2,1).contiguous(), ind) * weights.unsqueeze(1), dim=3)
        new_features = new_features.permute(0,2,1).contiguous().view(-1,point_features.shape[1]).contiguous()
        point_features = point_features.transpose(0,1) * (1-batch_dict['point_cls_scores']*0.2) + new_features.transpose(0,1) * batch_dict['point_cls_scores']*0.2
        point_features = point_features.contiguous().transpose(0,1).contiguous()
        batch_dict['point_features'] = point_features

        point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        ret_dict = {'point_cls_preds': point_cls_preds,
                    'point_box_preds': point_box_preds}
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']

        if not self.training or self.predict_boxes_when_training:
            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                points=batch_dict['point_coords'][:, 1:4],
                point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['point_coords'][:, 0]
            batch_dict['cls_preds_normalized'] = False

        self.forward_ret_dict = ret_dict

        return batch_dict
