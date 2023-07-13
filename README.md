# DEPICTER: Deep rEPresentation ClusTERing 

## Installation

We recommend creating a conda environment for running the whole DEPICTER pipeline:
```shell
conda env create -n depicter_env -f environment.yaml
```

To activate the environment:
```shell
conda activate depicter_env
```

## Patch extraction

The first step is to divide the whole slide images into patches and save them in TIF format under `/path/to/saving`. The images should be saved under `/path/to/images` and have their corresponding masks saved in `/path/to/masks`. Depending on the naming used on them, minor details might be needed in the code `extract.patches.py` code. 

The command bellow is an example run extracting patches from a magnification level corresponding to the second level of the pyramid of size 224 x 224 with no overlap and accepting only patches where the masks covers at least 90%. 

```shell
python extract_patches.py \
--slide_path='/path/to/images' \
--mask_path='/path/to/masks' \
--level=2 \
--patch_shape=224 \
--overlap=0 \
--mask_th=0.9 \
--save_path='/path/to/saving' \
```

## (basic) Embedding generation using a pretrained model

The next step is to generate the embedding

## (advanced) Pretraining your own self-supervised model
