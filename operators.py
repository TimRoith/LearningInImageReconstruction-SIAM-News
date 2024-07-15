import numpy as np
import skimage as ski
import torch
import torch.nn.functional as F
from torch.fft import fftfreq, fftshift, fft, ifft
from torch import Tensor
import math
from functools import partial

class Identity:
    def __call__(self,u):
        return u
    
    def adjoint(self, u):
        return u
    
    def inv(self, u):
        return u

class Radon:
    def __init__(self, theta=None, circle=True):
        self.theta = theta if not theta is None else np.linspace(0,180, 50)
        self.num_theta = len(self.theta)
        self.circle = circle
    
    def __call__(self, u):
        return ski.transform.radon(u, self.theta, circle=self.circle)/u.shape[-1]
    
    def adjoint(self, k):
        return ski.transform.iradon(k, self.theta, filter_name=None, circle=self.circle )/(k.shape[0] * np.pi/(2 * self.num_theta))
    
    def inv(self, k):
        return ski.transform.iradon(k * k.shape[0], self.theta, circle=self.circle)
    
    inverse = inv
    T = adjoint

class TV:
    def __init__(self,):
        self.grad = Grad()
    
    def __call__(self, u):
        return np.linalg.norm(self.grad(u).ravel(), ord=1)
    
def test_adjoint(A, x, y=None):
    Ax = A(x)
    if y is None:
        y = np.random.uniform(size=Ax.shape)
    res_1 = np.sum(Ax * y)
    res_2 = np.sum(x * A.adjoint(y))
    return res_1, res_2

class L1_norm:
    @staticmethod
    def __call__(u, lamda=1.):
        return lamda * torch.linalg.norm(u.ravel(), ord=1)

    @staticmethod
    def prox(u, lamda=1.):
        return soft_shrinkage(u, lamda)


class torch_radon(torch.autograd.Function):
    @staticmethod
    def forward(u, theta):
        u_np = u.cpu().detach().numpy()
        return torch.tensor(ski.transform.radon(u_np, theta.numpy())/u.shape[-1], dtype=u.dtype, device=u.device)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta = inputs
        ctx.save_for_backward(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta = ctx.saved_tensors[0]
        num_theta = len(theta)
        grad_output_np = grad_output.detach().numpy()
        grad_input = ski.transform.iradon(grad_output_np, theta.numpy(), filter_name=None,)/(grad_output_np.shape[0] * np.pi/(2 * num_theta))
        return torch.tensor(grad_input, dtype=grad_output.dtype, device=grad_output.device), None
    
class torch_radon_adjoint(torch.autograd.Function):
    @staticmethod
    def forward(k, theta):
        k_np = k.cpu().detach().numpy()
        num_theta = len(theta)
        u = ski.transform.iradon(k_np, theta.numpy(), filter_name=None,)/(k_np.shape[0] * np.pi/(2 * num_theta))
        return torch.tensor(u, dtype=k.dtype, device=k.device)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        _, theta = inputs
        ctx.save_for_backward(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta = ctx.saved_tensors[0]
        grad_output_np = grad_output.detach().numpy()
        return torch.tensor(ski.transform.radon(grad_output_np, theta.numpy())/u.shape[-1], dtype=u.dtype, device=u.device), None
    
    
def ramp_filter(size: int, device = None):
    # Create a ramp filter
    n = torch.cat((torch.arange(1, size / 2 + 1, 2, dtype=int),
                        torch.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = torch.zeros(size, device = device)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2

    ramp_filter = 2 * torch.fft.rfft(f)
    return ramp_filter

def _sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = (0, 0, pad_before, pad - pad_before,)
    return F.pad(sinogram, pad_width, mode='constant', value=0)

def _get_fourier_filter(size, filter_name, device='cpu'):
    """Construct the Fourier filter.

    This computation lessens artifacts and removes a small bias as
    explained in [1], Chap 3. Equation 61.

    Parameters
    ----------
    size : int
        filter size. Must be even.
    filter_name : str
        Filter used in frequency domain filtering. Filters available:
        ramp, shepp-logan, cosine, hamming, hann. Assign None to use
        no filter.

    Returns
    -------
    fourier_filter: ndarray
        The computed Fourier filter.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.

    """
    n = torch.concatenate(
        (
            torch.arange(1, size / 2 + 1, 2, dtype=int),
            torch.arange(size / 2 - 1, 0, -2, dtype=int),
        )
    )
    f = torch.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (torch.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * torch.real(fft(f))  # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = torch.pi * fftfreq(size)[1:]
        fourier_filter[1:] *= torch.sin(omega) / omega
    elif filter_name == "cosine":
        freq = torch.linspace(0, torch.pi, size, endpoint=False)
        cosine_filter = fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftshift(torch.hamming_window(size))
    elif filter_name == "hann":
        fourier_filter *= fftshift(torch.hann_window(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, None].to(device)



class Radon_torch:
    def __init__(self, theta, circle=True):
        self.theta = torch.arange(180) if theta is None else torch.tensor(theta)
        
    def __call__(self, u):
        return torch_radon.apply(u, self.theta)
        
    def adjoint(self, k):
        return iradon(k, self.theta, filter_name=None)/(k.shape[0] * torch.pi/(2 * len(self.theta)))
    
    def inv(self, k):
        return iradon(k * k.shape[0], self.theta, filter_name='ramp')

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.
    
    This function is copied from https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964 
    provided by user 0x00b1.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    oup_shape = x.shape
    x = x.view(-1)
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return (m[indicies] * x + b[indicies]).view(oup_shape)

def iradon(
    radon_image,
    theta=None,
    output_size=None,
    filter_name="ramp",
    interpolation="linear",
    circle=True,
    preserve_range=True,
):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    Parameters
    ----------
    radon_image : ndarray
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different
        angle. The tomography rotation axis should lie at the pixel
        index ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : array, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    output_size : int, optional
        Number of rows and columns in the reconstruction.
    filter_name : str, optional
        Filter used in frequency domain filtering. Ramp filter used by default.
        Filters available: ramp, shepp-logan, cosine, hamming, hann.
        Assign None to use no filter.
    interpolation : str, optional
        Interpolation method used in reconstruction. Methods available:
        'linear', 'nearest', and 'cubic' ('cubic' is slow).
    circle : boolean, optional
        Assume the reconstructed image is zero outside the inscribed circle.
        Also changes the default output_size to match the behaviour of
        ``radon`` called with ``circle=True``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    .. versionchanged:: 0.19
        In ``iradon``, ``filter`` argument is deprecated in favor of
        ``filter_name``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.

    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = torch.linspace(0, 180, radon_image.shape[1] + 1)[:-1]

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError(
            "The given ``theta`` does not match the number of "
            "projections in ``radon_image``."
        )

    interpolation_types = ('linear',)
    
    if interpolation not in interpolation_types:
        raise ValueError(f"Unknown interpolation: {interpolation}")

    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError(f"Unknown filter: {filter_name}")

    dtype = radon_image.dtype
    device = radon_image.device

    img_shape = radon_image.shape[0]
    if output_size is None:
        # If output size not specified, estimate from input radon image
        if circle:
            output_size = img_shape
        else:
            output_size = int(math.floor(math.sqrt((img_shape) ** 2 / 2.0)))

    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** math.ceil(math.log2(2 * img_shape))))
    pad_width = (0, 0, 0, projection_size_padded - img_shape)
    img = F.pad(radon_image, pad_width, mode='constant', value=0)

    # Apply filter in Fourier domain
    fourier_filter = _get_fourier_filter(projection_size_padded, filter_name, device=device)
    projection = fft(img, dim=0) * fourier_filter
    radon_filtered = torch.real(ifft(projection, dim=0)[:img_shape, :])
    
    # Reconstruct image by interpolation
    reconstructed = torch.zeros((output_size, output_size), dtype=dtype, device=device)
    radius = output_size // 2
    #xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    xpr, ypr = torch.meshgrid(torch.arange(output_size), torch.arange(output_size),  indexing='ij')
    xpr, ypr = (pr.to(device) - radius for pr in (xpr, ypr))        
    x = torch.arange(img_shape, device=device) - img_shape // 2

    for col, angle in zip(radon_filtered.T, torch.deg2rad(theta)):
        t = ypr * torch.cos(angle) - xpr * torch.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(interp, xp=x, fp=col,)
            #interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        else:
            pass
        reconstructed += interpolant(t)

    if circle:
        out_reconstruction_circle = (xpr**2 + ypr**2) > radius**2
        reconstructed[out_reconstruction_circle] = 0.0

    return reconstructed * torch.pi / (2 * angles_count)
        
    
    
def soft_shrinkage(x, lamda):
    return torch.clamp(torch.abs(x)-lamda, min=0.) * torch.sign(x)

class Identity:
    def __call__(self,u):
        return u
    
    def adjoint(self, u):
        return u
    
    def inv(self, u):
        return u

class TV:
    def __init__(self,):
        self.grad = Grad()
    
    def __call__(self, u):
        return torch.linalg.norm(self.grad(u).ravel(), ord=1)
    
def test_adjoint(A, x, y=None):
    Ax = A(x)
    if y is None:
        y = np.random.uniform(size=Ax.shape)
    res_1 = np.sum(Ax * y)
    res_2 = np.sum(x * A.adjoint(y))
    return res_1, res_2

class L1_norm:
    def __call__(self, u, lamda=1.):
        return lamda * np.linalg.norm(u.ravel(), ord=1)

    def prox(self, u, lamda=1.):
        return soft_shrinkage(u, lamda)



class Grad:
    def __call__(self, u):
        """
        applies a 2D image gradient to the image u of shape (n1,n2)
        
        Parameters
        ----------
        u : numpy 2D array, shape n1, n2
            Image

        Returns
        -------
        (px,py) image gradients in x- and y-directions.

        """
        n1 = u.shape[-2]
        n2 = u.shape[-1]
        px = torch.concatenate((u[1:,:]-u[0:-1,:], torch.zeros((1,n2), device=u.device)),axis=0)
        py = torch.concatenate((u[:,1:]-u[:,0:-1], torch.zeros((n1,1), device=u.device)),axis=1)
        return torch.concatenate((px[None,...],py[None,...]), axis=0)

    def adjoint(self, p):
        """
        Computes the negative divergence of the 2D vector field px,py.
        can also be seen as a tensor from R^(n1xn2x2) to R^(n1xn2)

        Parameters
        ----------
            - p : 2 x n1 x n2 torch.array

        Returns
        -------
            - divergence, n1 x n2 torch.array
        """
        u1 = torch.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
        u2 = torch.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
        return (u1+u2)
        
        
