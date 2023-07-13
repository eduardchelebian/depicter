import os
import glob
import argparse
from utils.patch_extraction import load_images, load_images_resize, generate_coordinates, save_patches 

def extract_patches(args):
    files = glob.glob(os.path.join(args.slide_path, '*'))
    print('Found ', len(files), ' files')
    print('------------------------------------------------------------------------------')
    for i, file in enumerate(files):
        slide_name=os.path.basename(file)
        print(f'Loading {os.path.basename(file)} ...    ({i})/{len(files)}')
        image, mask = load_images(os.path.join(args.slide_path, slide_name), os.path.join(args.mask_path, slide_name[:-4]+'_mask.tif'), args.level)
        coords = generate_coordinates(mask.shape, args.patch_shape, args.overlap)
        save_patches(coords, args.mask_th, image, mask, os.path.join(args.save_path, slide_name[:-4]))
        print(f'Saved patches for {os.path.basename(file)}     ({i})/{len(files)}')
        print('------------------------------------------------------------------------------')

# def extract_patches(args):
#     files = glob.glob(os.path.join(args.mask_path, '*'))
#     print('Found ', len(files), ' files')
#     print('------------------------------------------------------------------------------')
#     for i, file in enumerate(files):
#         file = os.path.basename(file)[:-4]
#         print(f'Loading {os.path.basename(file)} ...    ({i+1}/{len(files)})')
#         image, mask = load_images_resize(os.path.join(args.slide_path, file+'.tif'), 
#                                          os.path.join(args.mask_path, file+'.jpg'), 
#                                          args.level)
#         coords = generate_coordinates(mask.shape, args.patch_shape, args.overlap)
#         save_patches(coords, args.mask_th, image, mask, os.path.join(args.save_path, os.path.basename(file)))
#         print(f'Saved patches for {os.path.basename(file)}     ({i+1}/{len(files)})')
#         print('------------------------------------------------------------------------------')


# def extract_patches(args):
#     files = list(set(glob.glob(os.path.join(args.slide_path, '*'))) - set(glob.glob(os.path.join(args.slide_path, '*mask*'))))
#     print('Found ', len(files), ' files')
#     print('------------------------------------------------------------------------------')
#     for i, file in enumerate(files):
#         file = os.path.basename(file)[:-4]
#         print(f'Loading {os.path.basename(file)} ...    ({i+1}/{len(files)})')
#         image, mask = load_images(os.path.join(args.slide_path, file+'.jpg'), 
#                                          os.path.join(args.mask_path, file+'_tissuue_mask.jpg'), 
#                                          args.level)
#         coords = generate_coordinates(mask.shape, args.patch_shape, args.overlap)
#         save_patches(coords, args.mask_th, image, mask, os.path.join(args.save_path, os.path.basename(file)))
#         print(f'Saved patches for {os.path.basename(file)}     ({i+1}/{len(files)})')
#         print('------------------------------------------------------------------------------')

parser = argparse.ArgumentParser(description='Patch and feature extraction configuration')
parser.add_argument('--slide_path', type=str, default=None, 
                    help='path to slide image or directory')
parser.add_argument('--mask_path', type=str, default=None, 
                    help='path to mask image or directory')
parser.add_argument('--level', type=int, default=2, 
                    help='level of the pyramid to use')
parser.add_argument('--patch_shape', type=int, default=224, 
                    help='desired shape of the patches')
parser.add_argument('--overlap', type=float, default=.5, 
                    help='percentage of overlap between patches (0-1)')
parser.add_argument('--save_path', type=str, default=None, 
                    help='directory to save the extracted patches')
parser.add_argument('--mask_th', type=float, default=.5, 
                    help='minimum percentage of mask to accept a patch (0-1)')

args = parser.parse_args()
extract_patches(args)
print('Script finished!')