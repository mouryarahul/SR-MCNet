import math
import numpy as np

import torch
import torch.nn.functional as F

from typing import Union, Callable

from PIL import Image
from skimage.io import imread
from skimage.transform import resize, rescale

import cv2 as cv

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix


def cv2_resize(img_in: np.ndarray, scale: float, order: int):
    if img_in.ndim == 2:
        M,N = img_in.shape
    elif img_in.ndim == 3 and img_in.shape[2] == 3:
        M,N,C = img_in.shape
    else:
        raise Exception("Input image should be single or three color channels!")

    P,Q = (int(M*scale), int(N*scale))

    if order == 0:
       img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_NEAREST)
    elif order == 1:
        img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_LINEAR)
    elif order == 3:
        img_out = cv.resize(img_in, (Q, P), interpolation=cv.INTER_CUBIC)
    else:
        raise Exception("Unsupported Interpolation Order.")

    return img_out

def torch_resize(img_in: Union[np.ndarray, torch.Tensor], scale: float, order: int):
    ndim = img_in.ndim
    shape = img_in.shape
    dtype = type(img_in)
    if ndim == 2:
        if dtype == np.ndarray:
            img_in = torch.from_numpy(img_in).unsqueeze(dim=0).unsqueeze(dim=0)
        elif dtype == torch.Tensor:
            img_in = img_in.unsqueeze(dim=0).unsqueeze(dim=0)
    elif ndim == 3:
        if dtype == np.ndarray and shape[2] == 3: 
            img_in = torch.from_numpy(img_in.transpose(2,0,1)).unsqueeze(dim=0)
        elif dtype == torch.Tensor:
            if shape[2] == 3:
                img_in = img_in.permute(2,0,1).unsqueeze(dim=0)
            elif shape[1] == 3:
                img_in = img_in.unsqueeze(dim=0)
            else:
                raise TypeError("Input should be in either in 'HWC' or 'CHW'!")
        else:
            raise TypeError("Input should be in either NDArray or Tensor!")
    else:
        raise Exception("Input image should be single or three color channels!")
    
    if order == 0:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, mode='nearest')
    elif order == 1:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, antialias=True, mode='bilinear')
    elif order == 3:
        img_out = F.interpolate(img_in, size=None, scale_factor=scale, align_corners=False, antialias=True, mode='bicubic')
    else:
        raise Exception("Unsupported Interpolation Order.")
    
    img_out = img_out.squeeze()
    if dtype == np.ndarray:
        img_out = img_out.numpy()
        if ndim == 3:
            img_out = img_out.transpose(1,2,0)
    else:
       if ndim == 3 and shape[2] == 3:
            img_out = img_out.permute(1,2,0)
            
    return img_out


def pil_resize(img_in: np.ndarray, scale: float, order: int):
    if img_in.ndim == 2:
        M,N = img_in.shape
    elif img_in.ndim == 3 and img_in.shape[2] == 3:
        M,N,C = img_in.shape
    else:
        raise Exception("Input image should be single or three color channels!")
    
    P,Q = (int(M*scale), int(N*scale))
    if order == 0:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.NEAREST))
    elif order == 1:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.BILINEAR))
    elif order == 3:
        img_out = np.asarray(Image.fromarray(img_in).resize((Q, P), resample=Image.Resampling.BICUBIC))
    else:
        raise Exception("Unsupported Interpolation Order.")

    return img_out

def scikit_resize(img_in: np.ndarray, scale: float, order: int):
    if order == 0 or order == 1 or order == 3:
        img_out = rescale(img_in, scale, order, clip=False, preserve_range=True, anti_aliasing=True, channel_axis=2)
        return img_out
    else:
        raise Exception("Unsupported Interpolation Order.")


def generate_down_sampler(input_shape: tuple, scale: int, order: int, method: int):
    M, N = tuple(input_shape)
    P, Q = (M//scale, N//scale)
    n = M*N
    m = P*Q
    sampler = np.zeros((n, P, Q), dtype=np.float64)
    for i in range(n):
        row, col = divmod(i, N)  # row-major (C-style)
        ith_col_ident = np.zeros((M,N), dtype=np.float64)
        ith_col_ident[row, col] = 1.0
        
        if method == 1:
            if order == 0:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.NEAREST))
            elif order == 1:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BILINEAR))
            elif order == 3:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BICUBIC))
            else:
                raise Exception("Unsupported Interpolation Method.")
        elif method == 2:
            if order == 0:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_NEAREST)
            elif order == 1:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_LINEAR)
            elif order == 3:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_CUBIC)
            else:
                raise Exception("Unsupported Interpolation Method.")
        elif method == 3:
            sampler[i] = rescale(ith_col_ident, (1/scale), order, clip=False, preserve_range=False)
        elif method == 4:
            sampler[i] = torch_resize(ith_col_ident, (1/scale), order)
        else:
            raise Exception("Unsupported Interpolation Method.")

    sampler = sampler.reshape(n, m).astype(np.float32)
    return sampler.T

# @njit(nogil=True, cache=True)
def generate_up_sampler(input_shape: tuple, scale: int, order: int, method: int):
    M, N = tuple(input_shape)
    P, Q = (int(M*scale), int(N*scale))
    n = M*N
    m = P*Q
    sampler = np.zeros((n, P, Q), dtype=np.float64)
    for i in range(n):
        row, col = divmod(i, N)  # row-major (C-style)
        ith_col_ident = np.zeros((M,N), dtype=np.float64)
        ith_col_ident[row, col] = 1.0
        
        if method == 1:
            if order == 0:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.NEAREST))
            elif order == 1:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BILINEAR))
            elif order == 3:
                sampler[i] = np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BICUBIC))
            else:
                raise Exception("Unsupported Interpolation Order.")
        elif method == 2:
            if order == 0:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_NEAREST)
            elif order == 1:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_LINEAR)
            elif order == 3:
                sampler[i] = cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_CUBIC)
            else:
                raise Exception("Unsupported Interpolation Order.")
        elif method == 3:
            sampler[i] = rescale(ith_col_ident, scale, order, clip=False, preserve_range=False)
        elif method == 4:
            sampler[i] = torch_resize(ith_col_ident, scale, order)
        else:
            raise Exception("Unsupported Interpolation Method.")

    sampler = sampler.reshape(n, m).astype(np.float32)
    return sampler.T


def generate_down_sampler_sparse(input_shape: tuple, scale: int, order: int, method: int):
    M, N = tuple(input_shape)
    P, Q = (M//scale, N//scale)
    n = M*N
    m = P*Q
    sampler = lil_matrix((m, n), dtype=np.float32)
    for i in range(n):
        row, col = divmod(i, N)  # row-major (C-style)
        ith_col_ident = np.zeros((M,N), dtype=np.float32)
        ith_col_ident[row, col] = 1.0
        
        if method == 1:
            if order == 0:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.NEAREST)).reshape(m,1))
            elif order == 1:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BILINEAR)).reshape(m,1))
            elif order == 3:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BICUBIC)).reshape(m,1))
            else:
                raise Exception("Unsupported Interpolation Method.")
        elif method == 2:
            if order == 0:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_NEAREST).reshape(m,1))
            elif order == 1:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_LINEAR).reshape(m,1))
            elif order == 3:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_CUBIC).reshape(m,1))
            else:
                raise Exception("Unsupported Interpolation Method.")
        elif method == 3:
            sampler[:,i] = lil_matrix(rescale(ith_col_ident, (1/scale), order, clip=False, preserve_range=False).reshape(m,1))
        elif method == 4:
            sampler[:,i] = lil_matrix(torch_resize(ith_col_ident, (1/scale), order).reshape(m,1))
        else:
            raise Exception("Unsupported Interpolation Method.")

    return csr_matrix(sampler)


def generate_up_sampler_sparse(input_shape: tuple, scale: int, order: int, method: int):
    M, N = tuple(input_shape)
    P, Q = (int(M*scale), int(N*scale))
    n = M*N
    m = P*Q
    sampler = lil_matrix((m,n), dtype='float32')
    for i in range(n):
        row, col = divmod(i, N)  # row-major (C-style)
        ith_col_ident = np.zeros((M,N), dtype=np.float32)
        ith_col_ident[row, col] = 1.0
        
        if method == 1:
            if order == 0:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.NEAREST)).reshape(m,1))
            elif order == 1:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BILINEAR)).reshape(m,1))
            elif order == 3:
                sampler[:,i] = lil_matrix(np.asarray(Image.fromarray(ith_col_ident).resize((P,Q), resample=Image.Resampling.BICUBIC)).reshape(m,1))
            else:
                raise Exception("Unsupported Interpolation Order.")
        elif method == 2:
            if order == 0:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_NEAREST).reshape(m,1))
            elif order == 1:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_LINEAR).reshape(m,1))
            elif order == 3:
                sampler[:,i] = lil_matrix(cv.resize(ith_col_ident, (Q,P), interpolation=cv.INTER_CUBIC).reshape(m,1))
            else:
                raise Exception("Unsupported Interpolation Order.")
        elif method == 3:
            sampler[:,i] = lil_matrix(rescale(ith_col_ident, scale, order, clip=False, preserve_range=False).reshape(m,1))
        elif method == 4:
            sampler[:,i] = lil_matrix(torch_resize(ith_col_ident, scale, order).reshape(m,1))
        else:
            raise Exception("Unsupported Interpolation Method.")

    return csr_matrix(sampler)
