# DEPICTER: Deep rEPresentation ClusTERing 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Static Badge](https://img.shields.io/badge/demo-camelyon-brightgreen)](https://tissuumaps.scilifelab.se/patient_022_node_4.tmap?path=private/DEPICTER/camelyon)

## TO-DO
- [ ] Add images once preprint/article is available.
- [ ] Add citation links once preprint/article is available.
- [ ] Add "Advanced: Pretraining your own self-supervised model" section.

## Installation

### Utils
We recommend creating a conda environment for running the whole DEPICTER pipeline:
```shell
conda env create -n depicter_env -f environment.yml
```

To activate the environment:
```shell
conda activate depicter_env
```
### Interactive tool
In order to use the interactive tool, [installing TissUUmaps 3.0](https://tissuumaps.github.io/installation/) is also necessary. Please follow the instructions to install on Windows, Linux or macOS.

To install the DEPICTER plugin itself, start TissUUmaps (either the executable file or from terminal) and click on **Plugins** on the top left, **Add plugin**, tick the **DEPICTER** box and click **OK**. After restarting TissUUmaps, DEPICTER will appear in the **Plugins** tab.
 
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

## Using DEPICTER

1. The image you want to annotate in TissUUmaps by dragging it and dropping it.
2. Click on the **plus (+)** sign on the left channel and select the `[experiment].h5ad` file created for the corresponding file.
3. Select `/obsm/spatial;0` as **X coordinate** and `/obsm/spatial;1` as **Y coordinate**. Click **Update view**.
4. On the top left, select **Plugins** and **DEPICTER**. You may need to adjust the *Marker size* on the top right.
5. Place the *Positive class* (usually cancer) and *Negative class* seeds either by clicking on the markers or by holding shift and drawing around them.
6. Now you have two options:
    * Click **Run Seeded Iterative Clustering**. Correct and repeat as needed.
    * Based on where the positive seeds ended up in the feature space, click shift and draw around the markers in the feature space. Click **Feature space annotation** to complete the rest of the annotations with the negative class.
7. When you are happy with the results, they can be downloaded as CSV containing the (X, Y) coordinates the DEPICTER parametes and the final class.

## Advanced: Pretraining your own self-supervised model

We used [lightly](https://docs.lightly.ai/self-supervised-learning/index.html) for pretraining self-supervised models with each dataset. You can find the installation instructions [here](https://docs.lightly.ai/self-supervised-learning/getting_started/install.html).

Modyfing [lighly's SimCLR tutorial](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html), `pretrain_simclr.py` contains the hyperparameters used for fine-tuning every model, starting from the previously mentioned model by [Ciga et al. 2021](https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent). Note that we additionally used the [stainlib](https://github.com/sebastianffx/stainlib) library for H&E specific augmentations. The resulting collate function:

```
collate_fn = ImageCollateFunction(input_size = 224,
                                 min_scale = 0.25,
                                 vf_prob = 0.5,
                                 hf_prob = 0.5,
                                 rr_prob = 0.5,
                                 hed_thresh = 0.3)
```
