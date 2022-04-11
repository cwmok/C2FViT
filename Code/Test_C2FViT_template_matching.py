import os
from argparse import ArgumentParser

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from C2FViT_model import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise
from Functions import save_img, load_4D, min_max_norm


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelpath", type=str,
                        dest="modelpath",
                        default='../Model/C2FViT_affine_COM_template_matching_stagelvl3_116000.pth',
                        help="Pre-trained Model path")
    parser.add_argument("--savepath", type=str,
                        dest="savepath", default='../Result',
                        help="path for saving images")
    parser.add_argument("--fixed", type=str,
                        dest="fixed", default='../Data/MNI152_T1_1mm_brain_pad_RSP.nii.gz',
                        help="fixed image")
    parser.add_argument("--moving", type=str,
                        dest="moving", default='../Data/image_A.nii.gz',
                        help="moving image")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    savepath = opt.savepath
    fixed_path = opt.fixed
    moving_path = opt.moving
    com_initial = opt.com_initial
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).to(device)

    print(f"Loading model weight {opt.modelpath} ...")
    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()

    affine_transform = AffineCOMTransform().cuda()
    init_center = Center_of_mass_initial_pairwise()

    fixed_base = os.path.basename(fixed_path)
    moving_base = os.path.basename(moving_path)

    fixed_img_nii = nib.load(fixed_path)
    header, affine = fixed_img_nii.header, fixed_img_nii.affine
    fixed_img = fixed_img_nii.get_fdata()
    fixed_img = np.reshape(fixed_img, (1,) + fixed_img.shape)

    # If fixed img is MNI152 altas, do windowing (contrast stretching)
    if fixed_base == "MNI152_T1_1mm_brain_pad_RSP.nii.gz":
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

    moving_img = load_4D(moving_path)

    fixed_img = min_max_norm(fixed_img)
    moving_img = min_max_norm(moving_img)
    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        if com_initial:
            moving_img, init_flow = init_center(moving_img, fixed_img)

        X_down = F.interpolate(moving_img, scale_factor=0.5, mode="trilinear", align_corners=True)
        Y_down = F.interpolate(fixed_img, scale_factor=0.5, mode="trilinear", align_corners=True)

        warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
        X_Y, affine_matrix = affine_transform(moving_img, affine_para_list[-1])

        X_Y_cpu = X_Y.data.cpu().numpy()[0, 0, :, :, :]
        save_img(X_Y_cpu, f"{savepath}/warped_{moving_base}", header=header, affine=affine)

    print("Result saved to :", savepath)