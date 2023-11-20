import math
import numpy as np

from scipy.fftpack import dct, idct

import torch
from torch.nn.functional import pad

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        torch.nn.init.constant_(m.bias.data, 0.0)


def sp_noise(image: np.ndarray, prob: float):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if image.ndim == 2:
        black = 0
        white = 255            
    elif image.ndim == 3 and image.shape[2] == 3:
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')
    else:
        raise TypeError("Input image can only be Gray or RGB!")
    
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def dct2D(x:np.ndarray):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2D(x:np.ndarray):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')


def psf2otf_numpy(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf).astype(np.complex64)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def psf2otf_torch(h: torch.Tensor, outsize: tuple):
    if torch.all(h == 0):
        return torch.zeros_like(h)

    M,N = outsize
    P,Q = h.shape
    h = pad(h, (0,N-Q,0,M-P), mode='constant')
    h = torch.roll(h, shifts=(-math.floor(P/2),-math.floor(Q/2)), dims=(0,1))
    H = torch.fft.fft2(h)
    n_ops = torch.sum(h.numel() * torch.log2(torch.tensor(h.shape)))
    f = torch.finfo(H.dtype)
    tol = f.eps * n_ops
    if torch.all(h.imag.abs() < tol):
        H = H.real
    return H


def dx_fft(x:np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    psf = np.array([1, -1], dtype=np.float32).reshape(1,2)
    otf = psf2otf_numpy(psf, (M,N))
    return np.real(np.fft.ifft2(otf * np.fft.fft2(x)))

def dy_fft(x:np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    psf = np.array([1, -1], dtype=np.float32).reshape(2,1)
    otf = psf2otf_numpy(psf, (M,N))
    return np.real(np.fft.ifft2(otf * np.fft.fft2(x)))

def dtd_fft(out_shape: tuple):
    psf_h = np.array([1, -1], dtype=np.float32).reshape(1,2)
    psf_v = np.array([1, -1], dtype=np.float32).reshape(2,1)
    DTD = (np.square(np.abs(psf2otf_numpy(psf_h, out_shape))) + np.square(np.abs(psf2otf_numpy(psf_v, out_shape)))).astype(np.float32)
    return DTD

def forward_diff(x: np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M,N)
    dx = np.diff(x, axis=1, append=np.expand_dims(x[:,0], axis=1)).reshape(-1)
    dy = np.diff(x, axis=0, append=np.expand_dims(x[0,:], axis=0)).reshape(-1)
    return np.concatenate((dx, dy), axis=0)

def adjoint_diff(dz: np.ndarray, out_shape: tuple):
    M,N = out_shape
    n = M*N
    dx = dz[0:n].reshape(M, N)
    dy = dz[n:2*n].reshape(M, N)
    pre = np.expand_dims((dx[:,-1] - dx[:, 0]), axis=1)
    dtxy = np.concatenate((pre, -np.diff(dx, axis=1)), axis=1)
    pre = np.expand_dims((dy[-1,:] - dy[0,:]), axis=0)
    dtxy = dtxy + np.concatenate((pre, -np.diff(dy, axis=0)))
    return dtxy.reshape(-1)

def dx(x: np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    return np.diff(x, axis=1, append=np.expand_dims(x[:,0], axis=1)).reshape(-1)

def dy(x: np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    return np.diff(x, axis=0, append=np.expand_dims(x[0,:], axis=0)).reshape(-1)

# @njit(nogil=True, cache=True)
def generate_diff_matrices(input_shape:tuple):
    M, N = input_shape
    # Generate down sampling kernel
    first_row = np.zeros(N, dtype=np.float32)
    first_row[0] = -1.0; first_row[1] = 1.0
    dx = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dx[i, :] = np.roll(first_row, i)
    DX = np.zeros((M*N, M*N), dtype=np.float32)
    for i in range(M):
        DX[N*i:N*(i+1), N*i:N*(i+1)] = dx
    first_row = np.zeros(M*N, dtype=np.float32)
    first_row[0] = -1.0; first_row[N] = 1.0
    DY = np.zeros((M*N, M*N), dtype=np.float32)
    for i in range(M*N):
        DY[i, :] = np.roll(first_row, i)
    return DX, DY


def rgb_to_ycbcr(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    # using formulas from https://en.wikipedia.org/wiki/YCbCr
    # output = torch.Tensor(input.data.new(*input.size()), device=input.device)
    output = input.new(*input.size())
    output[:, 0, :, :] = 16   + (65.738/256.0)*input[:, 0, :, :]  + (129.057/256.0)*input[:, 1, :, :] + (25.064/256.0)*input[:, 2, :, :]
    output[:, 1, :, :] = 128  - (37.945/256.0)*input[:, 0, :, :]  - (74.494/256.0)*input[:, 1, :, :]  + (112.439/256.0) * input[:, 2, :, :]
    output[:, 2, :, :] = 128  + (112.439/256.0)*input[:, 0, :, :] - (94.154/256.0)*input[:, 1, :, :]  - (18.285/256.0)*input[:, 2, :, :]
    return output

def ycbcr_to_rgb(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    # using formulas from https://en.wikipedia.org/wiki/YCbCr
    # output = torch.Tensor(input.data.new(*input.size()), device=input.device)
    output = input.new(*input.size())
    output[:, 0, :, :] = (298.082/256.0)*input[:, 0, :, :]  + (408.583/256.0)*input[:, 2, :, :] - 222.921
    output[:, 1, :, :] = (298.082/256.0)*input[:, 0, :, :]  - (100.291/256.0)*input[:, 1, :, :]  - (208.120/256.0) * input[:, 2, :, :] + 135.576
    output[:, 2, :, :] = (298.082/256.0)*input[:, 0, :, :]  + (516.412/256.0)*input[:, 1, :, :]  - 276.836
    return output


def rgb_to_y(input):
    # input is mini-batch N x 3 x H x W of an RGB image
    # using formulas from https://en.wikipedia.org/wiki/YCbCr
    B,C,H,W = input.shape
    output = input.new(B,1,H,W)
    output[:, 0, :, :] = 16.0   + (65.738*input[:, 0, :, :]  + 129.057*input[:, 1, :, :] + 25.064*input[:, 2, :, :]) / 256.0
    return output


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)

        y = 16. + (65.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.stack((y, cb, cr), 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.stack((r, g, b), 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10.0 * torch.log10(1.0 / torch.mean((img1 - img2)**2))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
