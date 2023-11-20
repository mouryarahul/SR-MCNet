# This file contains script to generate patches from 2D gray or color image
import numpy as np
from typing import Union

from skimage.io import imread
from matplotlib import pyplot as plt


def get_patches(img: np.ndarray, patch_size: Union[int, tuple], stride: Union[int, tuple]):
    assert isinstance(img, np.ndarray) and 2 <= len(img.shape) <= 3, "img should be ndarray with either single or three channels!"

    if len(img.shape) == 2:
        H, W = img.shape
        C = 1
        img = np.expand_dims(img, axis=2)
    else:
        H, W, C = img.shape
    
    if isinstance(patch_size, int):
        h, w = patch_size, patch_size
    elif isinstance(patch_size, tuple):
        if len(patch_size) == 1:
            h, w = patch_size[0], patch_size[0]
        else:
            h, w = patch_size[0], patch_size[1]
    else:
        raise ValueError("patch_size should be a integer or tuple of two integers!")

    if isinstance(stride, int):
        sh, sw = stride, stride
    elif isinstance(stride, tuple):
        if len(stride) == 1:
            sh, sw = stride[0], stride[0]
        else:
            sh, sw = stride[0], stride[1]
    else:
        raise ValueError("stride should be a integer or tuple of two intergers!")
    
    if H < h or W < w:
        raise ValueError("patch_size is larger than image size!")

    # Generate horizontal indices
    col_indices = np.arange(0, W-w, sw)
    row_indices = np.arange(0, H-h, sh)

    if (row_indices[-1] + h) < H:
        last_row_index = H - h
        row_indices = np.concatenate((row_indices, [last_row_index]))

    if (col_indices[-1] + w) < W:
        last_col_index = W - w
        col_indices = np.concatenate((col_indices, [last_col_index]))


    # Total number of Patches
    num_patches = col_indices.size*row_indices.size
    patches = np.empty((num_patches, h, w, C), dtype=img.dtype)
    

    k = 0
    for i in row_indices:
        for j in col_indices:
            patches[k] = img[i:i+h, j:j+w]
            k += 1

    return patches, row_indices, col_indices
    

def reconstruct_from_patches_old(patches: np.ndarray, row_indices: np.ndarray, col_indices: np.ndarray):
    assert isinstance(patches, np.ndarray) and len(patches.shape) == 4, "patches should be ndarray of dimensions 4!"
    assert isinstance(col_indices, np.ndarray) and len(col_indices.shape) == 1, "col_indices should be ndarray of dimension 1!"
    assert isinstance(row_indices, np.ndarray) and len(row_indices.shape) == 1, "col_indices should be ndarray of dimension 1!"

    # Get single patch size
    n, h, w, c = patches.shape

    assert n == len(col_indices)*len(row_indices), "number of patches = len(col_indices)*len(row_indices) !"

    # Determine the resulting image size
    rows = row_indices[-1] + h
    cols = col_indices[-1] + w

    # Pre allocate the resulting image ndarray
    img = np.zeros((rows, cols, c), dtype=patches.dtype)

    # Assemble the resulting image
    k = 0
    for i in row_indices:
        for j in col_indices:
            img[i:i+h, j:j+w] = patches[k]
            k += 1
    
    if c == 1:
        img = np.squeeze(img, axis=2)
        
    return img


def reconstruct_from_patches(patches: np.ndarray, row_indices: np.ndarray, col_indices: np.ndarray):
    assert isinstance(patches, np.ndarray) and len(patches.shape) == 4, "patches should be ndarray of dimensions 4!"
    assert isinstance(col_indices, np.ndarray) and len(col_indices.shape) == 1, "col_indices should be ndarray of dimension 1!"
    assert isinstance(row_indices, np.ndarray) and len(row_indices.shape) == 1, "col_indices should be ndarray of dimension 1!"

    # Get single patch size
    n, h, w, c = patches.shape

    assert n == len(col_indices)*len(row_indices), "number of patches = len(col_indices)*len(row_indices) !"

    # Determine the resulting image size
    rows = row_indices[-1] + h
    cols = col_indices[-1] + w

    # Pre allocate the resulting image ndarray
    img = np.zeros((rows, cols, c), dtype=np.float32)
    weights = np.zeros((rows, cols, c), dtype=np.float32)

    # Assemble the resulting image and weights
    k = 0
    for i in row_indices:
        for j in col_indices:
            img[i:i+h, j:j+w] += patches[k]
            weights[i:i+h, j:j+w] += np.ones_like(patches[k])
            k += 1
    
    if c == 1:
        img = np.squeeze(img, axis=2)
        weights = np.squeeze(weights, axis=2)
        # Perform averaging operation
        img = img / weights
    elif c == 3:
        for i in range(c):
            img[:,:,i] = img[:,:,i] / weights[:,:,i]
    else:
        raise ValueError("Unsupported number of channels!")
    
    img = img.astype(patches.dtype)    

    return img


if __name__ == '__main__':

    # Read an image
    img = imread('Set5/woman.bmp', as_gray=False)

    # Extract patches from image
    patches, row_indices, col_indices = get_patches(img, 32, 24)

    # Assemble the patches
    img_recon = reconstruct_from_patches(patches, row_indices, col_indices)

    # Plot the images
    plt.figure()
    plt.imshow(img)

    plt.figure()
    plt.imshow(img_recon)

    plt.show()



