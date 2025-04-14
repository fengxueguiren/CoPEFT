# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn> Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.fine_tuning_modules.copeft import CoAdapter

class PointPillarIntermediateCoPeft(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateCoPeft, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],
                                  kernel_size=1)
        ###### fine tuning #####
        in_features1 = 64
        in_features2 = 384
        self.copeft_adapter1 = CoAdapter(in_features1)
        self.copeft_adapter2 = CoAdapter(in_features2)
        ########################

        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 3, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        lidar_pose = data_dict['lidar_pose']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'pairwise_t_matrix': pairwise_t_matrix}
            


        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict) # batch_dict['spatial_features'].shape=[4, 64, 200, 704], [number_of_vehicles, C, ]
        batch_dict['spatial_features'] = self.copeft_adapter1(batch_dict['spatial_features'])
        batch_dict = self.backbone(batch_dict)
        batch_dict['spatial_features_2d'] = self.copeft_adapter2(batch_dict['spatial_features_2d'])

        spatial_features_2d = batch_dict['spatial_features_2d'] # torch.Size([2, 384, 100, 352]), [batch_size, C,]


        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}
        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(spatial_features_2d)})
            
        return output_dict