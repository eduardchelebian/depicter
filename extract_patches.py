import os
import glob
import argparse
from utils.patch_extraction import load_images, generate_coordinates, save_patches 

def extract_patches(args):
    # Get a list of all image files in the specified directory
    files = glob.glob(os.path.join(args.slide_path, '*'))
    print('Found ', len(files), ' files')
    print('------------------------------------------------------------------------------')
    for file in files:
        # Load the slide and mask images
        print(f'Loading {os.path.basename(file)} ...')
        image, mask = load_images(file, os.path.join(args.mask_path, os.path.basename(file)), args.level)
        # Generate the coordinates of the patches
        coords = generate_coordinates(mask.shape, args.patch_shape, args.overlap)
        # Save the extracted patches
        save_patches(coords, args.mask_th, image, mask, os.path.join(args.save_path, os.path.basename(file)))
        print(f'Saved patches for {os.path.basename(file)}')
        print('------------------------------------------------------------------------------')

# Define the command-line arguments
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