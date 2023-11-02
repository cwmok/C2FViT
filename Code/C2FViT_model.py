import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act2 = act_layer()

    def forward(self, x, H, W, D):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dwconv(x, H, W, D)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool3d(7)
            self.sr = nn.Conv3d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, D):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, D))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=3, embed_dim=768, flatten=True):
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W, self.D = img_size[0] // stride, img_size[1] // stride, img_size[2] // stride
        self.num_patches = self.H * self.W * self.D
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        # self.norm = nn.LayerNorm(embed_dim)
        self.flatten = flatten

        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, D = x.shape
        if self.flatten:
            # BCHW -> BNC
            x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        x = self.act(x)

        return x, H, W, D


# From "Conditional Positional Encodings for Vision Transformers" by Chu et al., 2021
# https://github.com/Meituan-AutoML/Twins/blob/fa2f80e62794eaa55e2c1fbb41679a718ff642d9/segmentation/gvt.py
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1, k=3):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=k, stride=s, padding=k//2, bias=False, groups=embed_dim)
        self.s = s

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W, D)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class C2F_ViT_stage(nn.Module):
    def __init__(self, img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12, embed_dims=[256, 256, 256],
                 num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.Identity, depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size,
                                                  patch_size=patch_size[i],
                                                  stride=stride[i],
                                                  in_chans=2,
                                                  embed_dim=embed_dims[i])
            stage = nn.ModuleList([
                                         Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                               attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
                                               sr_ratio=sr_ratios[i], linear=linear) for _ in range(depths[i])])

            head = nn.Sequential(
            nn.Linear(embed_dims[i], embed_dims[i] // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims[i] // 2, num_classes, bias=False),
            nn.Tanh()
            )

            setattr(self, f"patch_embed_{i + 1}_xy", patch_embed)
            setattr(self, f"stage_{i + 1}", stage)
            setattr(self, f"head_{i+1}", head)

        for i in range(self.num_stages-1):
            squeeze = nn.Conv3d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=1, padding=1)
            setattr(self, f"squeeze_{i + 1}", squeeze)

        self.avg_pool = nn.AvgPool3d(2, 2)
        self.affine_transform = AffineCOMTransform()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def image_pyramid(self, x, level=3):
        out = [x]
        for i in range(level - 1):
            x = self.avg_pool(x)
            out.append(x)

        return out[::-1]

    def forward(self, x, y):
        B = x.shape[0]

        warpped_x_list = []
        affine_list = []

        x = self.image_pyramid(x, self.num_stages)
        y = self.image_pyramid(y, self.num_stages)

        for i in range(self.num_stages):
            if i == 0:
                xy = torch.cat([x[i], y[i]], dim=1)
            else:
                xy = torch.cat([warpped_x_list[i-1], y[i]], dim=1)

            patch_embed_xy = getattr(self, f"patch_embed_{i + 1}_xy")
            xy_patch_embed, H, W, D = patch_embed_xy(xy)

            if i > 0:
                xy_patch_embed = xy_patch_embed + xy_fea

            xy_fea = xy_patch_embed
            stage_block = getattr(self, f"stage_{i + 1}")
            for blk in stage_block:
                xy_fea = blk(xy_fea, H, W, D)

            head = getattr(self, f"head_{i + 1}")
            affine = head(xy_fea.mean(dim=1))
            affine_list.append(affine)

            if i < self.num_stages - 1:
                warpped_x, _ = self.affine_transform(x[i + 1], affine)
                warpped_x_list.append(warpped_x)

                xy_fea = xy_fea.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3)
                squeeze = getattr(self, f"squeeze_{i + 1}")
                xy_fea = squeeze(xy_fea).flatten(2).transpose(1, 2)
            else:
                warpped_x, _ = self.affine_transform(x[i], affine)
                warpped_x_list.append(warpped_x)

        return warpped_x_list, y, affine_list


class C2F_ViT_stage_pos(nn.Module):
    def __init__(self, img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12, embed_dims=[256, 256, 256],
                 num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.Identity, depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size,
                                                  patch_size=patch_size[i],
                                                  stride=stride[i],
                                                  in_chans=2,
                                                  embed_dim=embed_dims[i])
            stage = nn.ModuleList([
                                         Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                               attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
                                               sr_ratio=sr_ratios[i], linear=linear) for _ in range(depths[i])])

            head = nn.Sequential(
            nn.Linear(embed_dims[i], embed_dims[i] // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims[i] // 2, num_classes, bias=False),
            nn.Tanh()
            )

            i_imgsize = img_size//(2**(num_stages-1-i))
            pos_embed = nn.Parameter(torch.zeros(1, (i_imgsize//stride[i]) ** 3, embed_dims[i]))
            trunc_normal_(pos_embed, std=0.02)

            setattr(self, f"patch_embed_{i + 1}_xy", patch_embed)
            setattr(self, f"stage_{i + 1}", stage)
            setattr(self, f"head_{i + 1}", head)
            setattr(self, f"pos_embed_{i + 1}", pos_embed)

        for i in range(self.num_stages-1):
            squeeze = nn.Conv3d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=1, padding=1)
            setattr(self, f"squeeze_{i + 1}", squeeze)

        self.avg_pool = nn.AvgPool3d(2, 2)
        self.affine_transform = AffineCOMTransform()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def image_pyramid(self, x, level=3):
        out = [x]
        for i in range(level - 1):
            x = self.avg_pool(x)
            out.append(x)

        return out[::-1]

    def forward(self, x, y):
        B = x.shape[0]

        warpped_x_list = []
        affine_list = []

        x = self.image_pyramid(x, self.num_stages)
        y = self.image_pyramid(y, self.num_stages)

        for i in range(self.num_stages):
            if i == 0:
                xy = torch.cat([x[i], y[i]], dim=1)
            else:
                xy = torch.cat([warpped_x_list[i-1], y[i]], dim=1)

            patch_embed_xy = getattr(self, f"patch_embed_{i + 1}_xy")
            xy_patch_embed, H, W, D = patch_embed_xy(xy)

            # position embedding
            pos_embed = getattr(self, f"pos_embed_{i + 1}")
            xy_patch_embed = xy_patch_embed + pos_embed

            if i > 0:
                xy_patch_embed = xy_patch_embed + xy_fea

            xy_fea = xy_patch_embed
            stage_block = getattr(self, f"stage_{i + 1}")
            for blk in stage_block:
                xy_fea = blk(xy_fea, H, W, D)

            head = getattr(self, f"head_{i+1}")
            affine = head(xy_fea.mean(dim=1))
            affine_list.append(affine)

            if i < self.num_stages - 1:
                warpped_x, _ = self.affine_transform(x[i + 1], affine)
                warpped_x_list.append(warpped_x)

                xy_fea = xy_fea.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3)
                squeeze = getattr(self, f"squeeze_{i + 1}")
                xy_fea = squeeze(xy_fea).flatten(2).transpose(1, 2)
            else:
                warpped_x, _ = self.affine_transform(x[i], affine)
                warpped_x_list.append(warpped_x)

        return warpped_x_list, y, affine_list


class C2F_ViT_stage_peg(nn.Module):
    def __init__(self, img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12, embed_dims=[256, 256, 256],
                 num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.Identity, depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        for i in range(self.num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size,
                                                  patch_size=patch_size[i],
                                                  stride=stride[i],
                                                  in_chans=2,
                                                  embed_dim=embed_dims[i])
            stage = nn.ModuleList([
                                         Block(dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                               attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer,
                                               sr_ratio=sr_ratios[i], linear=linear) for _ in range(depths[i])])

            head = nn.Sequential(
            nn.Linear(embed_dims[i], embed_dims[i] // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims[i] // 2, num_classes, bias=False),
            nn.Tanh()
            )

            pos_cnn = PosCNN(embed_dims[i], embed_dims[i], k=patch_size[i])

            setattr(self, f"patch_embed_{i + 1}_xy", patch_embed)
            setattr(self, f"stage_{i + 1}", stage)
            setattr(self, f"head_{i + 1}", head)
            setattr(self, f"pos_cnn_{i + 1}", pos_cnn)

        for i in range(self.num_stages-1):
            squeeze = nn.Conv3d(embed_dims[i], embed_dims[i + 1], kernel_size=3, stride=1, padding=1)
            setattr(self, f"squeeze_{i + 1}", squeeze)

        self.avg_pool = nn.AvgPool3d(2, 2)
        self.affine_transform = AffineCOMTransform()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def image_pyramid(self, x, level=3):
        out = [x]
        for i in range(level - 1):
            x = self.avg_pool(x)
            out.append(x)

        return out[::-1]

    def forward(self, x, y):
        B = x.shape[0]

        warpped_x_list = []
        affine_list = []

        x = self.image_pyramid(x, self.num_stages)
        y = self.image_pyramid(y, self.num_stages)

        for i in range(self.num_stages):
            if i == 0:
                xy = torch.cat([x[i], y[i]], dim=1)
            else:
                xy = torch.cat([warpped_x_list[i - 1], y[i]], dim=1)

            patch_embed_xy = getattr(self, f"patch_embed_{i + 1}_xy")
            xy_patch_embed, H, W, D = patch_embed_xy(xy)

            if i > 0:
                xy_patch_embed = xy_patch_embed + xy_fea

            xy_fea = xy_patch_embed
            stage_block = getattr(self, f"stage_{i + 1}")
            for index, blk in enumerate(stage_block):
                xy_fea = blk(xy_fea, H, W, D)
                if index == 0:
                    pos_cnn = getattr(self, f"pos_cnn_{i + 1}")
                    xy_fea = pos_cnn(xy_fea, H, W, D)

            head = getattr(self, f"head_{i+1}")
            affine = head(xy_fea.mean(dim=1))
            affine_list.append(affine)

            if i < self.num_stages - 1:
                warpped_x, _ = self.affine_transform(x[i + 1], affine)
                warpped_x_list.append(warpped_x)

                xy_fea = xy_fea.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3)
                squeeze = getattr(self, f"squeeze_{i + 1}")
                xy_fea = squeeze(xy_fea).flatten(2).transpose(1, 2)
            else:
                warpped_x, _ = self.affine_transform(x[i], affine)
                warpped_x_list.append(warpped_x)

        return warpped_x_list, y, affine_list


class AffineCOMTransform(nn.Module):
    def __init__(self, use_com=True):
        super(AffineCOMTransform, self).__init__()

        self.translation_m = None
        self.rotation_x = None
        self.rotation_y = None
        self.rotation_z = None
        self.rotation_m = None
        self.shearing_m = None
        self.scaling_m = None

        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        self.use_com = use_com

    def forward(self, x, affine_para):
        # Matrix that register x to its center of mass
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)

        to_center_matrix = torch.eye(4).cuda()
        reversed_to_center_matrix = torch.eye(4).cuda()
        if self.use_com:
            x_sum = torch.sum(x)
            center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / x_sum
            center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / x_sum
            center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / x_sum

            to_center_matrix[0, 3] = center_mass_x
            to_center_matrix[1, 3] = center_mass_y
            to_center_matrix[2, 3] = center_mass_z
            reversed_to_center_matrix[0, 3] = -center_mass_x
            reversed_to_center_matrix[1, 3] = -center_mass_y
            reversed_to_center_matrix[2, 3] = -center_mass_z

        self.translation_m = torch.eye(4).cuda()
        self.rotation_x = torch.eye(4).cuda()
        self.rotation_y = torch.eye(4).cuda()
        self.rotation_z = torch.eye(4).cuda()
        self.rotation_m = torch.eye(4).cuda()
        self.shearing_m = torch.eye(4).cuda()
        self.scaling_m = torch.eye(4).cuda()

        trans_xyz = affine_para[0, 0:3]
        rotate_xyz = affine_para[0, 3:6] * math.pi
        shearing_xyz = affine_para[0, 6:9] * math.pi
        scaling_xyz = 1 + (affine_para[0, 9:12] * 0.5)

        self.translation_m[0, 3] = trans_xyz[0]
        self.translation_m[1, 3] = trans_xyz[1]
        self.translation_m[2, 3] = trans_xyz[2]
        self.scaling_m[0, 0] = scaling_xyz[0]
        self.scaling_m[1, 1] = scaling_xyz[1]
        self.scaling_m[2, 2] = scaling_xyz[2]

        self.rotation_x[1, 1] = torch.cos(rotate_xyz[0])
        self.rotation_x[1, 2] = -torch.sin(rotate_xyz[0])
        self.rotation_x[2, 1] = torch.sin(rotate_xyz[0])
        self.rotation_x[2, 2] = torch.cos(rotate_xyz[0])

        self.rotation_y[0, 0] = torch.cos(rotate_xyz[1])
        self.rotation_y[0, 2] = torch.sin(rotate_xyz[1])
        self.rotation_y[2, 0] = -torch.sin(rotate_xyz[1])
        self.rotation_y[2, 2] = torch.cos(rotate_xyz[1])

        self.rotation_z[0, 0] = torch.cos(rotate_xyz[2])
        self.rotation_z[0, 1] = -torch.sin(rotate_xyz[2])
        self.rotation_z[1, 0] = torch.sin(rotate_xyz[2])
        self.rotation_z[1, 1] = torch.cos(rotate_xyz[2])

        self.rotation_m = torch.mm(torch.mm(self.rotation_z, self.rotation_y), self.rotation_x)

        self.shearing_m[0, 1] = shearing_xyz[0]
        self.shearing_m[0, 2] = shearing_xyz[1]
        self.shearing_m[1, 2] = shearing_xyz[2]

        output_affine_m = torch.mm(to_center_matrix, torch.mm(self.shearing_m, torch.mm(self.scaling_m,
                                                                                        torch.mm(self.rotation_m,
                                                                                                 torch.mm(
                                                                                                     reversed_to_center_matrix,
                                                                                                     self.translation_m)))))
        grid = F.affine_grid(output_affine_m[0:3].unsqueeze(0), x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, output_affine_m[0:3].unsqueeze(0)


class DirectAffineTransform(nn.Module):
    def __init__(self):
        super(DirectAffineTransform, self).__init__()

        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

    def forward(self, x, affine_para):
        affine_matrix = affine_para.reshape(1, 3, 4) + self.id

        grid = F.affine_grid(affine_matrix, x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)

        return transformed_x, affine_matrix


class Center_of_mass_initial_pairwise(nn.Module):
    def __init__(self):
        super(Center_of_mass_initial_pairwise, self).__init__()
        self.id = torch.zeros((1, 3, 4)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        self.id[0, 2, 2] = 1

        self.to_center_matrix = torch.zeros((1, 3, 4)).cuda()
        self.to_center_matrix[0, 0, 0] = 1
        self.to_center_matrix[0, 1, 1] = 1
        self.to_center_matrix[0, 2, 2] = 1

    def forward(self, x, y):
        # center of mass of x -> center of mass of y
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)
        # mask = (x > 0).float()
        # mask_sum = torch.sum(mask)
        x_sum = torch.sum(x)
        x_center_mass_x = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0])/x_sum
        x_center_mass_y = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1])/x_sum
        x_center_mass_z = torch.sum(x.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2])/x_sum

        y_sum = torch.sum(y)
        y_center_mass_x = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 0]) / y_sum
        y_center_mass_y = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 1]) / y_sum
        y_center_mass_z = torch.sum(y.permute(0, 2, 3, 4, 1)[..., 0] * id_grid[..., 2]) / y_sum

        self.to_center_matrix[0, 0, 3] = x_center_mass_x - y_center_mass_x
        self.to_center_matrix[0, 1, 3] = x_center_mass_y - y_center_mass_y
        self.to_center_matrix[0, 2, 3] = x_center_mass_z - y_center_mass_z

        grid = F.affine_grid(self.to_center_matrix, x.shape, align_corners=True)
        transformed_image = F.grid_sample(x, grid, align_corners=True)

        # print(affine_para)
        # print(output_affine_m[0:3])

        return transformed_image, grid


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=7, eps=1e-5):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size_1d = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size_1d/2))
        J_sum = conv_fn(J, weight, padding=int(win_size_1d/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size_1d/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size_1d/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size_1d/2))

        # compute cross correltorch.
        win_size = win_size_1d**ndims
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3, kernel=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.kernel = kernel
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool3d(I, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=self.kernel, stride=2, padding=self.kernel//2, count_include_pad=False)

        return sum(total_NCC)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.eps = 1e-5

    def forward(self, input, target):
        N = target.size(0)

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2. * (intersection.sum(1) + self.eps) / (input_flat.sum(1) + target_flat.sum(1) + self.eps)
        loss = 1. - loss.sum() / N

        return loss


class MulticlassDiceLossVectorize(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self):
        super(MulticlassDiceLossVectorize, self).__init__()
        self.Diceloss = DiceLoss()
        self.eps = 1e-5

    def forward(self, input, target):
        N, C, H, W, D = input.shape
        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)

        intersection = input_flat * target_flat
        loss = 2. * (torch.sum(intersection, dim=-1) + self.eps) / (torch.sum(input_flat, dim=-1) + torch.sum(target_flat, dim=-1) + self.eps)
        loss = 1. - torch.mean(loss, dim=-1)

        return torch.mean(loss)
