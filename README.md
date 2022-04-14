# Affine Medical Image Registration with Coarse-to-Fine Vision Transformer (C2FViT)
This is the official Pytorch implementation of "Affine Medical Image Registration with Coarse-to-Fine Vision Transformer" (CVPR 2022), written by Tony C. W. Mok and Albert C. S. Chung.

![plot](./Figure/overview.png?raw=true)

## Prerequisites
- `Python 3.5.2+`
- `Pytorch 1.3.0 - 1.7.1`
- `NumPy`
- `NiBabel`

This code was tested with `Pytorch 1.7.1` and NVIDIA TITAN RTX GPU.

## Training and testing scripts
- `Train_C2FViT_pairwise.py`: Train a C2FViT model in an <u>unsupervised</u> manner for pairwise registration (Inter-subject registration).

- `Train_C2FViT_pairwise_semi.py`: Train a C2FViT model in an <u>semi-supervised</u> manner for pairwise registration (Inter-subject registration).

- `Train_C2FViT_template_matching.py`: Train a C2FViT model in an <u>unsupervised</u> manner for brain template-matching (MNI152 space).

- `Train_C2FViT_template_matching_semi.py`: Train a C2FViT model in an <u>semi-supervised</u> manner for brain template-matching (MNI152 space).

- `Test_C2FViT_template_matching.py`: Register an image pair with a pretrained C2FViT model (Template-matching).

- `Test_C2FViT_pairwise.py`: Register an image pair with a pretrained C2FViT model (Pairwise image registration).


## Inference
Template-matching (MNI152):

`python Test_C2FViT_template_matching.py --modelpath {model_path} --fixed ../Data/MNI152_T1_1mm_brain_pad_RSP.nii.gz --moving {moving_img_path}
`

Pairwise image registration:

`python Test_C2FViT_pairwise.py --modelpath {model_path} --fixed {fixed_img_path} --moving {moving_img_path}`


## Pre-trained model weights
Pre-trained model weights can be downloaded with the links below:

Unsupervised:
- [C2FViT_affine_COM_pairwise_stagelvl3_118000.pth](https://drive.google.com/file/d/1CQvyx96YBor9D7TWvvqHs6fuiJl-Jfay/view?usp=sharing)
- [C2FViT_affine_COM_template_matching_stagelvl3_116000.pth](https://drive.google.com/file/d/1uIItkfByyDYtxVxsjems_1HATRzcVCWX/view?usp=sharing)

Semi-supervised:
- [C2FViT_affine_COM_pairwise_semi_stagelvl3_95000.pth](https://drive.google.com/file/d/1T5JvXa3dCkFoFXNe5k7m3TDVn9AJv2_H/view?usp=sharing)
- [C2FViT_affine_COM_template_matching_semi_stagelvl3_130000.pth](https://drive.google.com/file/d/1bfh_jVOK5Ip2bBuTpCPYlQGFCMWpG_cb/view?usp=sharing)

## Train your own model
Step 0 (optional): Download the preprocessed OASIS dataset from https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md and place it under the `Data` folder.

Step 1: Replace `/PATH/TO/YOUR/DATA` with the path of your training data, e.g., `../Data/OASIS`, and make sure `imgs` and `labels` are properly loaded in the training script.

Step 2: Run `python {training_script}`, see "Training and testing scripts" for more details.

## Publication
If you find this repository useful, please cite:
- **Affine Medical Image Registration with Coarse-to-Fine Vision Transformer**  
[Tony C. W. Mok](https://cwmok.github.io/ "Tony C. W. Mok"), Albert C. S. Chung  
CVPR2022. [eprint arXiv:2203.15216](https://arxiv.org/abs/2203.15216)


## Acknowledgment
Some codes in this repository are modified from [PVT](https://github.com/whai362/PVT) and [ViT](https://github.com/lucidrains/vit-pytorch).
The MNI152 brain template is provided by the [FLIRT (FMRIB's Linear Image Registration Tool)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT#Template_Images).

###### Keywords
Keywords: Affine registration, Coarse-to-Fine Vision Transformer, 3D Vision Transformer