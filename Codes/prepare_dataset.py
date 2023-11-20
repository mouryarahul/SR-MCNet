# Script to generate Training Image Patches Dataset
import os
import argparse
import h5py
import numpy as np
from skimage.transform import resize, rescale
from skimage.io import imread
from PIL import Image
from matplotlib import pyplot as plt

from interpolation import cv2_resize, pil_resize, torch_resize
from utils import convert_rgb_to_y
from generate_patches import get_patches


def train(args):
    # Steps
    # 1. Get entries of all image file in the folder
    # 2. For each image, perform following steps:
    ##      1. Convert image to Ycbcr format and get only Y channel and call it Y_hr
    ##      2. Extract a given number of patches of the given size from Y_hr 
    ##      3. Downsample (by Bicubic Interpolation) Y_hr patches by the given scale and call them Y_lr 
    ##      4. Upsample (by Bicubic Interpolation) Y_lr patches by the given scale and call it Y_lr_up patches
    #
    # 3. Save Y_hr, Y_lr and Y_lr_up patches in a single HDF5 file format

    # Select Interpolation method
    if args.interp_order == "NEAREST":
        order = 0
    elif args.interp_order == "BILINEAR":
        order = 1
    elif args.interp_order == "BICUBIC":
        order = 3
    else:
        raise ValueError("Unsupported Interpolation method!")
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if not existed
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    with os.scandir(input_dir) as entries:
        bfirst_time: bool = True
        for entry in entries:
            if entry.is_file():
                # Read image file
                file_name = os.path.join(input_dir, entry.name)
                img = imread(file_name, as_gray=False)
                
                # Prepare it for Downsampling and Upsampling
                if img.ndim == 3 and img.shape[2] == 3:
                   img_height, img_width, img_channel = img.shape
                else:
                    raise Exception("Unsupported image file {}".format(file_name))

                img_width = (img_width // args.scale) * args.scale
                img_height = (img_height // args.scale) * args.scale
                img_hr = np.asarray(Image.fromarray(img).resize((img_width,img_height), resample=Image.Resampling.BICUBIC))
                
                if img_height < 80 or img_width < 80:
                    continue

                if args.interp_method == "PIL":
                    # print("Choosing PIL")
                    img_lr = pil_resize(img_hr, (1/args.scale), order)
                    img_lr_up = pil_resize(img_lr, args.scale, order)
                elif args.interp_method == "TORCH":
                    # print("Choosing TORCH")
                    img_lr = torch_resize(img_hr.astype(np.float32), (1/args.scale), order)
                    img_lr_up = torch_resize(img_lr.astype(np.float32), args.scale, order)
                elif args.interp_method == "CV":
                    # print("Choosing CV")
                    img_lr = cv2_resize(img_hr, (1/args.scale), order)
                    img_lr_up = cv2_resize(img_lr, args.scale, order)
                else:
                    raise Exception("Unsupported Interpolation Method!")

                # Clip the output to proper range because Bicubic Interpolation can produce out-of-range values
                img_hr = img_hr.astype(np.uint8)
                img_lr = np.clip(img_lr, a_min=0, a_max=255).astype(np.uint8)
                img_lr_up = np.clip(img_lr_up, a_min=0, a_max=255).astype(np.uint8)

                if args.patch_size >= 32:
                    # Extract patches
                    img_hr_patches, _, _ = get_patches(img_hr, patch_size=args.patch_size, stride=args.stride)
                    img_lr_patches, _, _ = get_patches(img_lr, patch_size=(args.patch_size // args.scale), stride=(args.stride // args.scale))
                    img_lr_up_patches, _, _ = get_patches(img_lr_up, patch_size=args.patch_size, stride=args.stride)
                else:
                    raise Exception("Patch size should be equal to or greater than 32!")
                
                # Data Augmentation
                mode = np.random.randint(1, high=4)
                if mode == 1:
                    patches_lr = np.flip(img_hr_patches, axis=2)
                    img_hr_patches = np.concatenate((img_hr_patches, patches_lr), axis=0)

                    patches_lr = np.flip(img_lr_patches, axis=2)
                    img_lr_patches = np.concatenate((img_lr_patches, patches_lr), axis=0)

                    patches_lr = np.flip(img_lr_up_patches, axis=2)
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_lr), axis=0)
                elif mode == 2:
                    patches_up = np.flip(img_hr_patches, axis=1)
                    img_hr_patches = np.concatenate((img_hr_patches, patches_up), axis=0)

                    patches_up = np.flip(img_lr_patches, axis=1)
                    img_lr_patches = np.concatenate((img_lr_patches, patches_up), axis=0)

                    patches_up = np.flip(img_lr_up_patches, axis=1)
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_up), axis=0)
                elif mode == 3:
                    patches_rot90 = np.rot90(img_hr_patches, k=1, axes=(1,2))
                    img_hr_patches = np.concatenate((img_hr_patches, patches_rot90), axis=0)

                    patches_rot90 = np.rot90(img_lr_patches, k=1, axes=(1,2))
                    img_lr_patches = np.concatenate((img_lr_patches, patches_rot90), axis=0)

                    patches_rot90 = np.rot90(img_lr_up_patches, k=1, axes=(1,2))
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_rot90), axis=0)

                # N = img_hr_patches.shape[0]
                # plt.figure()
                # plt.imshow(img_hr_patches[0].squeeze())
                # plt.figure()
                # plt.imshow(img_lr_patches[0].squeeze())
                # plt.figure()
                # plt.imshow(img_lr_up_patches[0].squeeze())

                # plt.figure()
                # plt.imshow(img_hr_patches[int(N/2)].squeeze())
                # plt.figure()
                # plt.imshow(img_lr_patches[int(N/2)].squeeze())
                # plt.figure()
                # plt.imshow(img_lr_up_patches[int(N/2)].squeeze())
                # plt.show()

                # Output file path
                tail = os.path.split(input_dir)[1]
                output_file_name = os.path.join(output_dir, "{}_{}_x{}.h5".format(tail, args.interp_order.lower(), args.scale))

                if bfirst_time:
                    bfirst_time = False
                    # create/open HDF5 file handle with write access
                    with h5py.File(output_file_name, 'w') as hf:
                        hf.create_dataset('hr', data=img_hr_patches, compression="gzip", chunks=True, maxshape=(None, img_hr_patches.shape[1], img_hr_patches.shape[2], img_hr_patches.shape[3]))
                        hf.create_dataset('lr', data=img_lr_patches, compression="gzip", chunks=True, maxshape=(None, img_lr_patches.shape[1], img_lr_patches.shape[2], img_lr_patches.shape[3]))
                        hf.create_dataset('lr_up', data=img_lr_up_patches, compression="gzip", chunks=True, maxshape=(None,img_lr_up_patches.shape[1], img_lr_up_patches.shape[2], img_lr_up_patches.shape[3]))
                        hf.close()
                else:
                    with h5py.File(output_file_name, 'a') as hf:
                        hf["hr"].resize((hf["hr"].shape[0] + img_hr_patches.shape[0]), axis = 0)
                        hf["hr"][-img_hr_patches.shape[0]:] = img_hr_patches

                        hf["lr"].resize((hf["lr"].shape[0] + img_lr_patches.shape[0]), axis = 0)
                        hf["lr"][-img_lr_patches.shape[0]:] = img_lr_patches

                        hf["lr_up"].resize((hf["lr_up"].shape[0] + img_lr_up_patches.shape[0]), axis = 0)
                        hf["lr_up"][-img_lr_up_patches.shape[0]:] = img_lr_up_patches
                        hf.close()

def train_new(args):
    # Steps
    # 1. Get entries of all image file in the folder
    # 2. For each image, perform following steps:
    ##      1. Convert image to Ycbcr format and get only Y channel and call it Y_hr
    ##      2. Extract a given number of patches of the given size from Y_hr 
    ##      3. Downsample (by Bicubic Interpolation) Y_hr patches by the given scale and call them Y_lr 
    ##      4. Upsample (by Bicubic Interpolation) Y_lr patches by the given scale and call it Y_lr_up patches
    #
    # 3. Save Y_hr, Y_lr and Y_lr_up patches in a single HDF5 file format

    # Select Interpolation method
    if args.interp_order == "NEAREST":
        order = 0
    elif args.interp_order == "BILINEAR":
        order = 1
    elif args.interp_order == "BICUBIC":
        order = 3
    else:
        raise ValueError("Unsupported Interpolation method!")
    
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if not existed
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Output file path
    tail = os.path.split(input_dir)[1]
    output_file_path = os.path.join(output_dir, "{}_{}_x{}.h5".format(tail, args.interp_order.lower(), args.scale))
    h5_file = h5py.File(output_file_path, 'w')

    lr_group = h5_file.create_group('lr')
    lr_up_group = h5_file.create_group('lr_up')
    hr_group = h5_file.create_group('hr')
    
    with os.scandir(input_dir) as entries:
        patch_idx = 0
        for entry in entries:
            if entry.is_file():
                # Read image file
                file_name = os.path.join(input_dir, entry.name)
                img = imread(file_name, as_gray=False)
                
                # Prepare it for Downsampling and Upsampling
                if img.ndim == 3 and img.shape[2] == 3:
                   img_height, img_width, img_channel = img.shape
                else:
                    raise Exception("Unsupported image file {}".format(file_name))

                img_width = (img_width // args.scale) * args.scale
                img_height = (img_height // args.scale) * args.scale
                img_hr = np.asarray(Image.fromarray(img).resize((img_width,img_height), resample=Image.Resampling.BICUBIC))
                
                if img_height < 80 or img_width < 80:
                    continue

                if args.interp_method == "PIL":
                    # print("Choosing PIL")
                    img_lr = pil_resize(img_hr, (1/args.scale), order)
                    img_lr_up = pil_resize(img_lr, args.scale, order)
                elif args.interp_method == "TORCH":
                    # print("Choosing TORCH")
                    img_lr = torch_resize(img_hr.astype(np.float32), (1/args.scale), order)
                    img_lr_up = torch_resize(img_lr.astype(np.float32), args.scale, order)
                elif args.interp_method == "CV":
                    # print("Choosing CV")
                    img_lr = cv2_resize(img_hr, (1/args.scale), order)
                    img_lr_up = cv2_resize(img_lr, args.scale, order)
                else:
                    raise Exception("Unsupported Interpolation Method!")

                # Clip the output to proper range because Bicubic Interpolation can produce out-of-range values
                img_hr = img_hr.astype(np.uint8)
                img_lr = np.clip(img_lr, a_min=0, a_max=255).astype(np.uint8)
                img_lr_up = np.clip(img_lr_up, a_min=0, a_max=255).astype(np.uint8)

                if args.patch_size >= 32:
                    # Extract patches
                    img_hr_patches, _, _ = get_patches(img_hr, patch_size=args.patch_size, stride=args.stride)
                    img_lr_patches, _, _ = get_patches(img_lr, patch_size=(args.patch_size // args.scale), stride=(args.stride // args.scale))
                    img_lr_up_patches, _, _ = get_patches(img_lr_up, patch_size=args.patch_size, stride=args.stride)
                else:
                    raise Exception("Patch size should be equal to or greater than 32!")
                
                # Data Augmentation
                mode = np.random.randint(1, high=4)
                if mode == 1:
                    patches_lr = np.flip(img_hr_patches, axis=2)
                    img_hr_patches = np.concatenate((img_hr_patches, patches_lr), axis=0)

                    patches_lr = np.flip(img_lr_patches, axis=2)
                    img_lr_patches = np.concatenate((img_lr_patches, patches_lr), axis=0)

                    patches_lr = np.flip(img_lr_up_patches, axis=2)
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_lr), axis=0)
                elif mode == 2:
                    patches_up = np.flip(img_hr_patches, axis=1)
                    img_hr_patches = np.concatenate((img_hr_patches, patches_up), axis=0)

                    patches_up = np.flip(img_lr_patches, axis=1)
                    img_lr_patches = np.concatenate((img_lr_patches, patches_up), axis=0)

                    patches_up = np.flip(img_lr_up_patches, axis=1)
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_up), axis=0)
                elif mode == 3:
                    patches_rot90 = np.rot90(img_hr_patches, k=1, axes=(1,2))
                    img_hr_patches = np.concatenate((img_hr_patches, patches_rot90), axis=0)

                    patches_rot90 = np.rot90(img_lr_patches, k=1, axes=(1,2))
                    img_lr_patches = np.concatenate((img_lr_patches, patches_rot90), axis=0)

                    patches_rot90 = np.rot90(img_lr_up_patches, k=1, axes=(1,2))
                    img_lr_up_patches = np.concatenate((img_lr_up_patches, patches_rot90), axis=0)

                N = img_hr_patches.shape[0]
                for i in range(N):
                    lr_group.create_dataset(str(patch_idx), data=img_lr_patches[i], compression="gzip")
                    lr_up_group.create_dataset(str(patch_idx), data=img_lr_up_patches[i], compression="gzip")
                    hr_group.create_dataset(str(patch_idx), data=img_hr_patches[i], compression="gzip")
                    patch_idx += 1

    h5_file.close()

                
def eval(args):
    # Select Interpolation method
    if args.interp_order == "NEAREST":
        order = 0
    elif args.interp_order == "BILINEAR":
        order = 1
    elif args.interp_order == "BICUBIC":
        order = 3
    else:
        raise ValueError("Unsupported Interpolation method!")

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Create output directory if it do not exist already
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Output file path
    tail = os.path.split(input_dir)[1]
    output_file_name = os.path.join(output_dir, "{}_{}_x{}.h5".format(tail, args.interp_order.lower(), args.scale))

    hf = h5py.File(output_file_name, 'w')
    
    lr_group = hf.create_group('lr')
    hr_group = hf.create_group('hr')
    lr_up_group = hf.create_group('lr_up')      
    
    with os.scandir(input_dir) as entries:
        for i, entry in enumerate(entries):
            if entry.is_file():
                # Read image file
                file_name = os.path.join(input_dir, entry.name)
                img = imread(file_name, as_gray=False)

                # Prepare it for Downsampling and Upsampling
                if img.ndim == 3 and img.shape[2] == 3:
                   img_height, img_width, img_channel = img.shape
                else:
                    raise Exception("Unsupported image file {}".format(file_name))
                
                img_width = (img_width // args.scale) * args.scale
                img_height = (img_height // args.scale) * args.scale
                img_hr = np.asarray(Image.fromarray(img).resize((img_width,img_height), resample=Image.Resampling.BICUBIC))

                if img_height < 80 or img_width < 80:
                    continue

                # Perform Up and then Down sampling
                if args.interp_method == "PIL":
                    # print("Choosing PIL")
                    img_lr = pil_resize(img_hr, (1/args.scale), order)
                    img_lr_up = pil_resize(img_lr, args.scale, order)
                elif args.interp_method == "TORCH":
                    # print("Choosing TORCH")
                    img_lr = torch_resize(img_hr.astype(np.float32), (1/args.scale), order)
                    img_lr_up = torch_resize(img_lr.astype(np.float32), args.scale, order)
                elif args.interp_method == "CV":
                    # print("Choosing CV")
                    img_lr = cv2_resize(img_hr, (1/args.scale), order)
                    img_lr_up = cv2_resize(img_lr, args.scale, order)
                else:
                    raise Exception("Unsupported Interpolation Method!")

                # Clip the output to proper range because Bicubic Interpolation can produce out-of-range value
                img_hr = img_hr.astype(np.uint8)
                img_lr = np.clip(img_lr, a_min=0, a_max=255).astype(np.uint8)
                img_lr_up = np.clip(img_lr_up, a_min=0, a_max=255).astype(np.uint8)

                lr_group.create_dataset(str(i), data=img_lr)
                hr_group.create_dataset(str(i), data=img_hr)
                lr_up_group.create_dataset(str(i), data=img_lr_up)

        hf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='../Data/TrainImages/T91')
    parser.add_argument('--output-dir', type=str, default='../Data/datasets_TORCH')
    parser.add_argument('--stride', type=int, default=24)  # with T91 chose 24, with T100 choose 30
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--interp-order', type=str, default="BICUBIC")
    parser.add_argument('--interp-method', type=str, default="TORCH")
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    parser.add_argument('--patch-size', type=int, default=20*args.scale)
    args = parser.parse_args()

    if not args.eval:
        print("Generating Training Dataset for scale {}...".format(args.scale))
        train_new(args)
    else:
        print("Generating Evaluation Dataset for scale {}...".format(args.scale))
        eval(args)
