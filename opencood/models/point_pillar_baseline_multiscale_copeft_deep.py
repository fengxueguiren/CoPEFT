# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Support F-Cooper, Self-Att, DiscoNet(wo KD), V2VNet, V2XViT, When2comm

import torch.nn as nn
import torch
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone 
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.fusion_in_one import AttFusion_deep
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.models.fine_tuning_modules.coprompt import CoPrompt
from opencood.models.fuse_modules.fusion_in_one import regroup

class PointPillarBaselineMultiscaleCoPEFTDeep(nn.Module):
    print('### PointPillarBaselineMultiscaleCoPEFTDeep')
    def __init__(self, args):
        super(PointPillarBaselineMultiscaleCoPEFTDeep, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        is_resnet = args['base_bev_backbone'].get("resnet", True) # default true
        if is_resnet:
            self.backbone = ResNetBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        else:
            self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64) # or you can use ResNetBEVBackbone, which is stronger
        self.voxel_size = args['voxel_size']

        self.fusion_net = nn.ModuleList()
        for i in range(len(args['base_bev_backbone']['layer_nums'])):
            # if args['fusion_method'] == "max":
            #     self.fusion_net.append(MaxFusion())
            if args['fusion_method'] == "att":
                self.fusion_net.append(AttFusion_deep(args['att']['feat_dim'][i]))
        self.out_channel = sum(args['base_bev_backbone']['num_upsample_filter'])

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            self.out_channel = args['shrink_header']['dim'][-1]

        self.compression = False
        if "compression" in args:
            self.compression = True
            self.naive_compressor = NaiveCompressor(64, args['compression'])

        self.cls_head = nn.Conv2d(self.out_channel, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.out_channel, 7 * args['anchor_number'],
                                  kernel_size=1)
        ###### fine tuning #####
        # in_features1 = 64
        # in_features2 = 256
        # size = [1, 64, 200, 504]
        self.max_cav = args['max_cav']
        self.copeft_vehicle_prompt = CoPrompt(hidden_size=64)

        self.copeft_vehicle_prompt_inner0 = CoPrompt(hidden_size=64)
        self.copeft_vehicle_prompt_inner1 = CoPrompt(hidden_size=128)
        self.copeft_vehicle_prompt_inner2 = CoPrompt(hidden_size=256)

        self.copeft_vehicle_prompt_inner_list = [self.copeft_vehicle_prompt_inner0, self.copeft_vehicle_prompt_inner1, self.copeft_vehicle_prompt_inner2]

        self.copeft_vehicle_prompt_late = CoPrompt(hidden_size=256)
        
        ########################

        self.use_dir = False
        if 'dir_args' in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(self.out_channel, args['dir_args']['num_bins'] * args['anchor_number'],
                                  kernel_size=1) # BIN_NUM = 2
 
        if 'backbone_fix' in args.keys() and args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        """
        Fix the parameters of backbone during finetune on timedelay。
        """
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False
        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        for p in self.cls_head.parameters():
            p.requires_grad = False
        for p in self.reg_head.parameters():
            p.requires_grad = False

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}
        # n, 4 -> n, c
        batch_dict = self.pillar_vfe(batch_dict)
        # n, c -> N, C, H, W
        batch_dict = self.scatter(batch_dict)
        # calculate pairwise affine transformation matrix
        _, _, H0, W0 = batch_dict['spatial_features'].shape # original feature map shape H0, W0
        normalized_affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']

        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)
        
        ################################## adapter and prompt tuning
        # generate prompt
        spatial_features, vehicle_prompt, prompt_num = self.copeft_vehicle_prompt(spatial_features, record_len)

        # add prompt to spatial_features
        spatial_features_regroup = regroup(spatial_features, record_len)
        spatial_features_regroup_list = list()
        prompt_num_list = list()
        for index, spatial_features_item in enumerate(spatial_features_regroup):
            if spatial_features_item.shape[0] >= self.max_cav:
                print(f"########spatial_features_item.shape[0]={spatial_features_item.shape[0]}")
                spatial_features_regroup_list.append(spatial_features_item)
                prompt_num_list.append(0)
            elif prompt_num == 1:
                spatial_features_regroup_list.append(torch.cat((spatial_features_item, vehicle_prompt[index]), dim=0))
                prompt_num_list.append(prompt_num)
            else:
                pass

            # if prompt_num == 1:
            #     spatial_features_regroup_list.append(torch.cat((spatial_features_item, vehicle_prompt[index]), dim=0))
            # else:
            #     pass

        spatial_features = torch.cat(spatial_features_regroup_list, dim=0)

        # update record_len by number prompt_num
        prompt_num = torch.tensor(prompt_num_list).to(record_len.device)
        record_len = record_len + prompt_num
        ######################

        # multiscale fusion
        feature_list = self.backbone.get_multiscale_feature(spatial_features)
        fused_feature_list = []
        # adapter inner
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(fuse_module(feature_list[i], record_len, normalized_affine_matrix, self.copeft_vehicle_prompt_inner_list[i]))
        fused_feature = self.backbone.decode_multiscale_feature(fused_feature_list) 

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
        # # adapter 2
        fused_feature = self.copeft_vehicle_prompt_late(fused_feature, record_len-prompt_num, only_adapter=True)

        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)

        output_dict = {'cls_preds': psm,
                       'reg_preds': rm}

        if self.use_dir:
            output_dict.update({'dir_preds': self.dir_head(fused_feature)})

        return output_dict
