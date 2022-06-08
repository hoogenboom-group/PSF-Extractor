import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    'gaussian_1D',
    'gaussian_2D',
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


def fit_gaussian_1D(y, x=None, p0=None):
    """Fit a 1D Gaussian"""
    if x is None:
        x = np.arange(y.size)

    if p0 is None:
        p0 = guess_gaussian_1D_params(y, x)
    popt, _ = curve_fit(gaussian_1D, x, y, p0=p0)
    return popt

def fit_gaussian_2D(image, p0=None, theta=None, epsilon=1e-3):
    """Fit a 2D Gaussian

    Parameters
    ----------
    image : array-like
    p0 : list of initial fitting estimates
        Default : None
    theta : float or 2-valued tuple
        Determines whether an elliptical 2D Guassian is fitted.
        float : fixates the angle during fitting
        2-valued tuple : set the lower and upper bounds during fitting
        None : default value, fit normal 2D Gaussian
    epsilon : float
        Theta fitting range

    References
    ----------
    [1] https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
    """
    # TODO: check robustness parameter estimation 
    # and fitting for data with high intensity outliers
    sy, sx = image.shape
    
    # Make a meshgrid in the shape of the image
    x = np.arange(sx)
    y = np.arange(sy)
    xx, yy = np.meshgrid(x, y)

    # Ravel the meshgrids of X, Y points to a pair of 1D arrays
    X = np.stack((xx.ravel(), yy.ravel()), axis=0)
    # Ravel the image data
    image_1D = image.ravel()

    # Trick `curve_fit` into fitting 2D data
    if p0 is None:  # make a crude initial guess for parameters if not provided
        p0 = guess_gaussian_2D_params(image)
     
    # Add bounds to maintain sanity     
    fit_bounds = ([0, 0, 0, 0, 0, 0], 
          [sx, sy, sx, sy, image.max(), image.max()])
    if theta is None:
        fit_func = _gaussian_2D
    else:
        fit_func = _elliptical_gaussian_2D
        if type(theta) is tuple:
            # TODO make sure tuple length is 2
            for i in range(2): fit_bounds[i].insert(4, theta[i])
        else:
            for i in range(2): fit_bounds[i].insert(4, theta+(2*(i-0.5)*epsilon))
        p0 = list(p0)
        p0.insert(4, np.mean(theta))
    popt, _ = curve_fit(fit_func, X, image_1D, p0=p0, 
                        bounds=fit_bounds)
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


def _elliptical_gaussian_2D(M, *args):
    """Wrapper for `elliptical_gaussian_2D` to pass 
    to `scipy.optimize.curve_fit`

    References
    ----------
    [1] https://scipython.com/blog/non-linear-least-squares-fitting-of-a-two-dimensional-data/
    """
    # TODO: don't like to have two almost identical 
    # wrapper functions
    # M = array([[ 0,  1,  2, ..., N-2, N-1, N], --> x
    #            [ 0,  0,  0, ..., N, N, N]])    --> y
    x, y = M
    return elliptical_gaussian_2D(x, y, *args)

def guess_gaussian_1D_params(y, x=None):
    """Make initial estimates for a 1D Gaussian fit"""
    # Create vector in x if not provided
    if x is None:
        x = np.arange(y.size)

    # Estimate x0
    x0 = x[y.argmax()]
    # Estimate sigma_x
    xs = x[y > (y.max() + y.min())/2]  # threshold at half-maximum
    # sigma = FWHM / (2*sqrt(2*ln(2)))
    sigma_x = (xs.max() - xs.min()) / 2.355
    # Estimates for amplitude and background
    A = y.max() - y.min()
    B = y.min()
    return x0, sigma_x, A, B

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
