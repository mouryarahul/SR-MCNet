import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix

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


def psf2otf(psf, shape):
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
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def dx_fft(x:np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    psf = np.array([1, -1], dtype=np.float32).reshape(1,2)
    otf = psf2otf(psf, (M,N))
    return np.real(np.fft.ifft2(otf * np.fft.fft2(x)))


def dy_fft(x:np.ndarray, out_shape: tuple):
    M,N = out_shape
    x = x.reshape(M, N)
    psf = np.array([1, -1], dtype=np.float32).reshape(2,1)
    otf = psf2otf(psf, (M,N))
    return np.real(np.fft.ifft2(otf * np.fft.fft2(x)))


def dtd_fft(out_shape: tuple):
    psf_h = np.array([1, -1], dtype=np.float32).reshape(1,2)
    psf_v = np.array([1, -1], dtype=np.float32).reshape(2,1)
    DTD = (np.square(np.abs(psf2otf(psf_h, out_shape))) + np.square(np.abs(psf2otf(psf_v, out_shape)))).astype(np.float32)
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


def generate_diff_matrices_dense(input_shape:tuple):
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


def generate_diff_matrices_sparse(input_shape:tuple):
    M, N = input_shape
    # Generate down sampling kernel
    first_row = np.zeros(N, dtype=np.float32)
    first_row[0] = -1.0; first_row[1] = 1.0
    dx = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        dx[i, :] = np.roll(first_row, i)
    DX = lil_matrix((M*N, M*N), dtype=np.float32)
    for i in range(M):
        DX[N*i:N*(i+1), N*i:N*(i+1)] = lil_matrix(dx)

    first_row = np.zeros(M*N, dtype=np.float32)
    first_row[0] = -1.0; first_row[N] = 1.0
    DY = lil_matrix((M*N, M*N), dtype=np.float32)
    for i in range(M*N):
        DY[i, :] = lil_matrix(np.roll(first_row, i))

    return csr_matrix(DX), csr_matrix(DY)
