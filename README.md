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

The command below is an example run extracting patches from a magnification level corresponding to the second level of the pyramid of size 224 x 224 with no overlap and accepting only patches where the masks covers at least 90%. 

```shell
python extract_patches.py \
--slide_path='/path/to/images' \
--mask_path='/path/to/masks' \
--level=2 \
--patch_shape=224 \
--overlap=0 \
--mask_th=0.9 \
--save_path='/path/to/saving'
```



## Embedding generation using a pretrained model

> Note: Networks pretrained on ImageNet or publicly available pretrained networks, such as the one proposed by [Ciga et al, 2021](https://doi.org/10.1016/j.mlwa.2021.100198) available [here](https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent) already showed great results with DEPICTER. Thus, we recommend this basic approach before going on trying to train your own model (advanced block explained afterwards).

After the images are divided into patches, we can generate their embeddings. This will produce, among others, the `[experiment].h5ad` file that will be the input to the DEPICTER plugin in TissUUmaps.

The command below is an example run for generating patch embeddings using a pretrained model with the ResNet18 architecture. One could include the `--imagenet` argument instead of `--no_imagenet` and `--model_path` to use the default weights pretrained on ImageNet.

```shell
python generate_embeddings.py \
--save_path='/path/to/saving'  \
--architecture='resnet18' \
--experiment_name='experiment' \
--no_imagenet \
--model_path='/path/to/pretrained/model.ckpt' \
--num_workers=32
```

## Advanced: Pretraining your own self-supervised model
