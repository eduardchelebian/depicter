import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tifffile import tifffile

Image.MAX_IMAGE_PIXELS = None



def load_images(slide_path, mask_path, level):
    """
    Load slide and mask images from TIF files using the tifffile library.

    Args:
        slide_path (str): The path to the slide image file.
        mask_path (str): The path to the mask image file.
        level (int): The level of the pyramid to use.

    Returns:
        A tuple containing the slide image and its mask, both as NumPy arrays.
    """

    try: 
        image = tifffile.imread(slide_path, level=level)
        mask = tifffile.imread(mask_path, level=level)

    except tifffile.TiffFileError:
        print('Not a TIFF file; loading conventionally ignoring `level` argument')
        image = Image.open(slide_path)
        mask = Image.open(mask_path)
        

    except IndexError:
        print('Could not load pyramid level; loading full image and resizing instead')

        image = tifffile.imread(slide_path)
        image = Image.fromarray(image)
        image = image.resize((image.width//(level**2), image.height//(level**2)))        

        mask = tifffile.imread(mask_path)
        mask = Image.fromarray(mask)
        mask = mask.resize((mask.width//(level**2), mask.height//(level**2)))        
    
    image = np.array(image)
    mask = np.array(mask)

    mask = np.asarray(mask) > 0
    mask = mask.astype(bool)
        
    return image, mask



def generate_coordinates(image_shape, patch_shape, overlap):
    """
    Generates a list of coordinates of patches to extract from an image.

    Args:
        image_shape (tuple): The shape of the image as a tuple of (height, width).
        patch_shape (int): The desired shape of the patches (assuming square patches).
        overlap (float): The percentage of overlap between patches (0-1).

    Returns:
        A list of lists, where each inner list contains the x start, x end, y start, and y end coordinates of a patch.
    """

    patch_shape = (patch_shape, patch_shape)
    stride = (patch_shape[0]*(1-overlap), patch_shape[1]*(1-overlap))
    xmax = image_shape[0] - patch_shape[0] 
    ymax = image_shape[1] - patch_shape[1]
    coords=[]
    step = np.ceil(np.divide(image_shape,stride)).astype(np.uint32)
    x = np.ceil(np.linspace(0, xmax, step[0])).astype(np.uint32)
    y = np.ceil(np.linspace(0, ymax, step[1])).astype(np.uint32)

    for i in range(x.size):
            for j in range(y.size):
                    xs = x[i]
                    xe = xs + patch_shape[0]

                    ys = y[j]
                    ye = ys + patch_shape[1]
                    coords.append([xs, xe, ys, ye])

    return coords


def save_patches(coords, mask_th, image, mask, save_dir):
    '''
    Saves patches defined by the generated grid of coordinates

    Args:
    - coords (list): a list of tuples representing the coordinates of the patches
    - mask_th (float): minimum percentage of mask to accept a patch (0-1)
    - image (np.array): image from which to extract patches
    - mask (boolean): tissue mask of the regions of interest
    - save_dir (str): path to patch saving directory

    Returns: None
    '''

    os.makedirs(save_dir, exist_ok=True)
    patch_shape = (coords[0][1] - coords[0][0], coords[0][3] - coords[0][2])
    pos_x = []
    pos_y = []

    print('Saving patches...')
    for i, indices in enumerate(tqdm(coords)):
        mask_patch = mask[indices[0]:indices[1], indices[2]:indices[3]]
        if np.mean(mask_patch) > mask_th:
            patch = image[indices[0]:indices[1], indices[2]:indices[3], :]
            patch = Image.fromarray(patch).convert('RGB')
            pos_x.append(indices[0] + patch_shape[0] / 2)
            pos_y.append(indices[2] + patch_shape[1] / 2)
            patch.save(
                os.path.join(
                    save_dir, 'y' + str(int(indices[0] + patch_shape[0] / 2)) + '_x' + str(
                        int(indices[2] + patch_shape[1] / 2)) + '.tif'
                )
            )

    pd.DataFrame(np.array([pos_y, pos_x]).transpose(),
                    columns=['x', 'y']).to_csv(os.path.join(save_dir, 'coords.csv'))
