import os
import glob
import argparse
from utils.embedding_generation import load_model, generate_umap, csv2h5ad

import torch

def generate_embeddings(args):
    print(torch.cuda.is_available())
    files = glob.glob(os.path.join(args.save_path, '*'))
    print('Found ', len(files), ' files')
    print('------------------------------------------------------------------------------')
    backbone = load_model(args.model_path, args.architecture, args.imagenet)
    for i, file in enumerate(files):
        print(f'Generating embeddings for {os.path.basename(file)} ...    ({i+1}/{len(files)})')
        generate_umap(backbone, os.path.join(args.save_path, os.path.basename(file)), args.num_workers, args.experiment_name)
        csv2h5ad(os.path.join(args.save_path, os.path.basename(file)), args.experiment_name)
        print(f'Embeddings generated for {os.path.basename(file)}    ({i+1}/{len(files)})')
        print('------------------------------------------------------------------------------')
        

parser = argparse.ArgumentParser(description='Feature extraction configuration')
parser.add_argument('--save_path', type=str, default=None, 
                    help='directory to save the extracted patches')
parser.add_argument('--architecture', type=str, default='resnet18', 
                    help='Model architecture (default: resnet18)')
parser.add_argument('--model_path', type=str, default=None, 
                    help='Path to model weights')
parser.add_argument('--imagenet', action='store_true')
parser.add_argument('--no_imagenet', dest='imagenet', action='store_false')
parser.add_argument('--num_workers', type=int, default=32, 
                    help='Number of workers')
parser.add_argument('--experiment_name', type=str, default=None, 
                    help='Name of the experiment')

args = parser.parse_args()
print(args)
generate_embeddings(args)
print('Script finished!')