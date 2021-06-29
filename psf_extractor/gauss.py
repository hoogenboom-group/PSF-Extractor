import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    'gaussian_1D',
    'gaussian_2D',
    'super_gaussian_1D',
    'elliptical_gaussian_2D',
    'super_elliptical_gaussian_2D',
    'fit_gaussian_1D',
    'fit_gaussian_2D',
    'guess_gaussian_1D_params',
    'guess_gaussian_2D_params'
]


def gaussian_1D(x, x0, sigma_x, A, B):
    """1D Gaussian function with added background"""
    E = (x - x0)**2 / (2*sigma_x**2)
    return A * np.exp(-E) + B


def gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, A, B):
    """2D Gaussian function with added background"""
    E = (x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)
    return A * np.exp(-E) + B


def super_gaussian_1D(x, x0, sigma_x, P, A, B):
    """Super (higher order) Gaussian function

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function#Higher-order_Gaussian_or_super-Gaussian_function
    """
    # E = (2**(2*P - 1)) * np.log(2) * ((x - x0)**2 / sigma_x**2)
    E = (x - x0)**2 / (2*sigma_x**2)
    return A * np.exp(-E**P)**2 + B


def elliptical_gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, theta, A, B):
    """Elliptical Gaussian function with added background

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """
    # Convert angle from degrees to radians
    theta = np.deg2rad(theta)
    a = np.cos(theta)**2 / (2*sigma_x**2) + np.sin(theta)**2 / (2*sigma_y**2)
    b = -np.sin(2*theta) / (4*sigma_x**2) + np.sin(2*theta) / (4*sigma_y**2)
    c = np.sin(theta)**2 / (2*sigma_x**2) + np.cos(theta)**2 / (2*sigma_y**2)
    return A * np.exp( -(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2)) + B


def super_elliptical_gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, P, A, B):
    """Higher order elliptical Gaussian function with added background

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function#Higher-order_Gaussian_or_super-Gaussian_function
    """
    E = (x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)
    return A * np.exp(-E**P) + B


def fit_gaussian_1D(y, x=None, p0=None):
    """Fit a 1D Gaussian"""
    if x is None:
        x = np.arange(y.size)

    if p0 is None:
        p0 = guess_gaussian_1D_params(y, x)
    popt, _ = curve_fit(gaussian_1D, x, y, p0=p0)
    return popt


def guess_gaussian_1D_params(y, x=None):
    """Make initial estimates for a 1D Gaussian fit"""
    # Create vector in x if not provided
    if x is None:
        x = np.arange(y.size)

    # Estimate x0
    x0 = x.mean()
    # Estimate sigma_x
    xs = x[y > y.max()/2]  # threshold at half-maximum
    # sigma = FWHM / (2*sqrt(2*ln(2))) ~ FWHM / 2.355
    sigma_x = (xs.max() - xs.min()) / 2.355
    # Estimates for amplitude and background
    A = y.max()
    B = y.min()
    return x0, sigma_x, A, B


def fit_gaussian_2D(image, p0=None):
    """Fit a 2D Gaussian

    References
    ----------
    [1] https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/"""
    # Make a meshgrid in the shape of the image
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    xx, yy = np.meshgrid(x, y)

    # Ravel the meshgrids of X, Y points to a pair of 1D arrays
    X = np.stack((xx.ravel(), yy.ravel()), axis=0)
    # Ravel the image data
    image_1D = image.ravel()

    # Trick `curve_fit` into fitting 2D data
    if p0 is None:  # make a crude initial guess for parameters if not provided
        p0 = guess_gaussian_2D_params(image)
    popt, _ = curve_fit(_gaussian_2D, X, image_1D, p0=p0)
    return popt


def _gaussian_2D(M, *args):
    """Wrapper for `gaussian_2D` to pass to `scipy.optimize.curve_fit`

    References
    ----------
    [1] https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
    """
    # M = array([[ 0,  1,  2, ..., N-2, N-1, N], --> x
    #            [ 0,  0,  0, ..., N, N, N]])    --> y
    x, y = M
    return gaussian_2D(x, y, *args)


def guess_gaussian_2D_params(image):
    """Make initial estimates for a 2D Gaussian fit"""
    # Mask the central spot
    im = image.copy()
    hm = im < im.max()/2  # threshold at half-maximum
    im[hm] = 0
    xy = np.argwhere(im > 0)
    ys, xs = xy[:, 0], xy[:, 1]

    # Make estimates for x0, y0
    x0, y0 = xs.mean(), ys.mean()

    # Make estimates for sigma_x, sigma_y
    # sigma = FWHM / (2*sqrt(2*ln(2))) ~ FWHM / 2.355
    sigma_x = (xs.max() - xs.min()) / 2.355
    sigma_y = (ys.max() - ys.min()) / 2.355

    # Estimates for amplitude and background
    A = image.max()
    B = image.mean()

    return x0, y0, sigma_x, sigma_y, A, B
