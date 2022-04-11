import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from C2FViT_model import C2F_ViT_stage, AffineCOMTransform, Center_of_mass_initial_pairwise, multi_resolution_NCC
from Functions import Dataset_epoch


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice / num_count


def train():
    print("Training C2FViT...")
    model = C2F_ViT_stage(img_size=128, patch_size=[3, 7, 15], stride=[2, 4, 8], num_classes=12,
                          embed_dims=[256, 256, 256],
                          num_heads=[2, 2, 2], mlp_ratios=[2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                          attn_drop_rate=0., norm_layer=nn.Identity,
                          depths=[4, 4, 4], sr_ratios=[1, 1, 1], num_stages=3, linear=False).cuda()

    # model = C2F_ViT_stage(img_size=128, patch_size=[7, 15], stride=[4, 8], num_classes=12, embed_dims=[256, 256],
    #                       num_heads=[2, 2], mlp_ratios=[2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #                       attn_drop_rate=0., norm_layer=nn.Identity, depths=[4, 4], sr_ratios=[1, 1], num_stages=2,
    #                       linear=False).cuda()

    # model = C2F_ViT_stage(img_size=128, patch_size=[15], stride=[8], num_classes=12, embed_dims=[256],
    #                       num_heads=[2], mlp_ratios=[2], qkv_bias=False, qk_scale=None, drop_rate=0.,
    #                       attn_drop_rate=0., norm_layer=nn.Identity, depths=[4], sr_ratios=[1], num_stages=1,
    #                       linear=False).cuda()

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    affine_transform = AffineCOMTransform().cuda()
    init_center = Center_of_mass_initial_pairwise()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)

    # OASIS
    imgs = sorted(glob.glob(datapath + "/OASIS_OAS1_*_MR1/norm.nii.gz"))
    labels = sorted(glob.glob(datapath + "/OASIS_OAS1_*_MR1/seg35.nii.gz"))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model/' + model_name[0:-1]

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    lossall = np.zeros((2, iteration + 1))

    training_generator = Data.DataLoader(Dataset_epoch(imgs, labels, norm=True, use_label=False),
                                         batch_size=1,
                                         shuffle=True, num_workers=4)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration:
        for X, Y in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()

            # COM initialization
            if com_initial:
                X, _ = init_center(X, Y)

            X = F.interpolate(X, scale_factor=0.5, mode="trilinear", align_corners=True)
            Y = F.interpolate(Y, scale_factor=0.5, mode="trilinear", align_corners=True)

            warpped_x_list, y_list, affine_para_list = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(warpped_x_list[-1], y_list[-1])

            loss = loss_multiNCC

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}"'.format(
                    step, loss.item(), loss_multiNCC.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Put your validation code here
                # ---------------------------------------

                # imgs = sorted(glob.glob(datapath + "/OASIS_OAS1_*_MR1/norm.nii.gz"))[255:259]
                # labels = sorted(glob.glob(datapath + "/OASIS_OAS1_*_MR1/seg35.nii.gz"))[255:259]
                #
                # valid_generator = Data.DataLoader(
                #     Dataset_epoch(imgs, labels, norm=True, use_label=True),
                #     batch_size=1,
                #     shuffle=False, num_workers=2)
                #
                # use_cuda = True
                # device = torch.device("cuda" if use_cuda else "cpu")
                # dice_total = []
                # brain_dice_total = []
                # print("\nValiding...")
                # for batch_idx, data in enumerate(valid_generator):
                #     X, Y, X_label, Y_label = data[0].to(device), data[1].to(device), data[2].to(
                #         device), data[3].to(device)
                #
                #     with torch.no_grad():
                #         if com_initial:
                #             X, init_flow = init_center(X, Y)
                #             X_label = F.grid_sample(X_label, init_flow, mode="nearest", align_corners=True)
                #
                #         X_down = F.interpolate(X, scale_factor=0.5, mode="trilinear", align_corners=True)
                #         Y_down = F.interpolate(Y, scale_factor=0.5, mode="trilinear", align_corners=True)
                #
                #         warpped_x_list, y_list, affine_para_list = model(X_down, Y_down)
                #         X_Y, affine_matrix = affine_transform(X, affine_para_list[-1])
                #         F_X_Y = F.affine_grid(affine_matrix, X_label.shape, align_corners=True)
                #
                #         X_Y_label = F.grid_sample(X_label, F_X_Y, mode="nearest", align_corners=True).cpu().numpy()[0,
                #                     0, :, :, :]
                #         X_brain_label = (X_Y > 0).float().cpu().numpy()[0, 0, :, :, :]
                #
                #         # brain mask
                #         Y_brain_label = (Y > 0).float().cpu().numpy()[0, 0, :, :, :]
                #         Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]
                #
                #         dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                #         dice_total.append(dice_score)
                #
                #         brain_dice = dice(np.floor(X_brain_label), np.floor(Y_brain_label))
                #         brain_dice_total.append(brain_dice)
                #
                # dice_total = np.array(dice_total)
                # brain_dice_total = np.array(brain_dice_total)
                # print("Dice mean: ", dice_total.mean())
                # print("Brain Dice mean: ", brain_dice_total.mean())
                #
                # with open(log_dir, "a") as log:
                #     log.write(f"{step}: {dice_total.mean()}, {brain_dice_total.mean()} \n")

            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--modelname", type=str,
                        dest="modelname",
                        default='C2FViT_affine_COM_pairwise_',
                        help="Model name")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--iteration", type=int,
                        dest="iteration", default=160001,
                        help="number of total iterations")
    parser.add_argument("--checkpoint", type=int,
                        dest="checkpoint", default=1000,
                        help="frequency of saving models")
    parser.add_argument("--datapath", type=str,
                        dest="datapath",
                        default='/PATH/TO/YOUR/DATA',
                        help="data path for training images")
    parser.add_argument("--com_initial", type=bool,
                        dest="com_initial", default=True,
                        help="True: Enable Center of Mass initialization, False: Disable")
    opt = parser.parse_args()

    lr = opt.lr
    iteration = opt.iteration
    n_checkpoint = opt.checkpoint
    datapath = opt.datapath
    com_initial = opt.com_initial

    model_name = opt.modelname

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

    print("Training %s ..." % model_name)
    train()


