import numpy as np
import itertools

import nibabel as nib
import numpy as np
import torch
import torch.utils.data as Data
import csv
import torch.nn.functional as F


def load_4D(name):
    # X = sitk.GetArrayFromImage(sitk.ReadImage(name, sitk.sitkFloat32 ))
    # X = np.reshape(X, (1,)+ X.shape)
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_4D_channel(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.transpose(X, (3, 0, 1, 2))
    return X


def min_max_norm(img):
    max = np.max(img)
    min = np.min(img)

    norm_img = (img - min) / (max - min)

    return norm_img


def save_img(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_flow(I_img, savename, header=None, affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, names, labels, norm=True, use_label=False):
        'Initialization'
        self.names = names
        self.labels = labels
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        self.index_pair_label = list(itertools.permutations(labels, 2))
        self.use_label = use_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        img_A_label = load_4D(self.index_pair_label[step][0])
        img_B_label = load_4D(self.index_pair_label[step][1])

        if self.norm:
            img_A = min_max_norm(img_A)
            img_B = min_max_norm(img_B)

        if self.use_label:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(img_A_label).float(), torch.from_numpy(img_B_label).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch_MNI152(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, label_list, fixed_img, fixed_label, need_label=True):
        'Initialization'
        super(Dataset_epoch_MNI152, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.label_pair = label_list
        self.need_label = need_label
        self.fixed_img = fixed_img
        self.fixed_label = fixed_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.img_pair[step])
        fixed_img = load_4D(self.fixed_img)
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

        if self.need_label:
            moving_label = load_4D(self.label_pair[step])
            fixed_label = load_4D(self.fixed_label)
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float(), torch.from_numpy(moving_label).float(), torch.from_numpy(fixed_label).float()
        else:
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float()


class Dataset_epoch_MNI152_pre_one_hot(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, img_list, label_list, fixed_img, fixed_label, need_label=True):
        'Initialization'
        super(Dataset_epoch_MNI152_pre_one_hot, self).__init__()
        # self.exp_path = exp_path
        self.img_pair = img_list
        self.label_pair = label_list
        self.need_label = need_label
        self.fixed_img = fixed_img
        self.fixed_label = fixed_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        moving_img = load_4D(self.img_pair[step])
        fixed_img = load_4D(self.fixed_img)
        fixed_img = np.clip(fixed_img, a_min=2500, a_max=np.max(fixed_img))

        if self.need_label:
            moving_label = load_4D_channel(self.label_pair[step])
            fixed_label = load_4D_channel(self.fixed_label)

            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float(), torch.from_numpy(moving_label).float(), torch.from_numpy(fixed_label).float()
        else:
            return torch.from_numpy(min_max_norm(moving_img)).float(), torch.from_numpy(
                min_max_norm(fixed_img)).float()


class Dataset_epoch_onehot(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, names, labels, norm=True, use_label=False):
        'Initialization'
        self.names = names
        self.labels = labels
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))
        self.index_pair_label = list(itertools.permutations(labels, 2))
        self.use_label = use_label

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        img_A_label = load_4D_channel(self.index_pair_label[step][0])
        img_B_label = load_4D_channel(self.index_pair_label[step][1])

        if self.norm:
            img_A = min_max_norm(img_A)
            img_B = min_max_norm(img_B)

        if self.use_label:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(img_A_label).float(), torch.from_numpy(img_B_label).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()
