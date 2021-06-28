import numpy as np
from scipy.optimize import curve_fit


__all__ = [
    'gaussian_1D',
    'gaussian_2D',
    'elliptical_gaussian_2D',
    'super_gaussian_1D',
    'super_elliptical_gaussian_2D',
    '',
    ''
]


def gaussian_1D(x, x0, sigma_x, A, B):
    """1D Gaussian function with added background"""
    E = (x - x0)**2 / (2*sigma_x**2)
    return A * np.exp(-E) + B


def gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, A, B):
    """2D Gaussian function with added background"""
    E = (x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)
    return A * np.exp(-E) + B


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


def super_gaussian_1D(x, x0, sigma_x, P, A, B):
    """Super (higher order) Gaussian function

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function#Higher-order_Gaussian_or_super-Gaussian_function
    """
    # E = (2**(2*P - 1)) * np.log(2) * ((x - x0)**2 / sigma_x**2)
    E = (x - x0)**2 / (2*sigma_x**2)
    return A * np.exp(-E**P)**2 + B


def super_elliptical_gaussian_2D(x, y, x0, y0, sigma_x, sigma_y, P, A, B):
    """Higher order elliptical Gaussian function with added background

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gaussian_function#Higher-order_Gaussian_or_super-Gaussian_function
    """
    E = (x-x0)**2/(2*sigma_x**2) + (y-y0)**2/(2*sigma_y**2)
    return A * np.exp(-E**P) + B


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
    popt, pcov = curve_fit(_gaussian_2D, X, image_1D, p0=p0)
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





# # This is the callable that is passed to curve_fit. M is a (2,N) array
# # where N is the total number of data points in Z, which will be ravelled
# # to one dimension.
# def _gaussian_2D(M, *args):
#     x, y = M
#     arr = np.zeros(x.shape)
#     for i in range(len(args)//7):
#         arr += gaussian_2D(x, y, *args[i*7:i*7+7])
#     return arr

# def do_2D_gauss_fit(arr, thetaest=45):
#     arry, arrx = arr.shape
#     midx, midy, sigx, sigy, maxI, minI = gauss2D_param(arr)
#     p0 = [midx, midy, sigx/4, sigy, thetaest, maxI, minI]
#     x, y = np.arange(0, arrx), np.arange(0, arry)
#     X, Y = np.meshgrid(x, y)
#     xdata = np.vstack((X.ravel(), Y.ravel()))
#     popt, pcov = curve_fit(_gaussian_2D, xdata, arr.ravel(), p0, maxfev = 8000)
#     return popt #give back all fit values


# def gauss2D_param(im): #estimate first guesses for parameters
#     imy, imx = im.shape
#     for fact in [3, 2.5, 2, 1.5, 1, 0.5]:
#         try:
#             image = im.copy()
#             idxs = image < image.mean() + fact*image.std()
#             idxs = scipy.ndimage.binary_dilation(idxs)
#             image[idxs] = 0
#             xy = np.argwhere(image > 0)
#             ys, xs = xy[:,0], xy[:,1]
#             midy, midx = ys.mean(), xs.mean()
#             sigy, sigx = (ys.max() - ys.min())/2, (xs.max() - xs.min())/2
#             yn, yp = intround(midy-sigy), intround(midy+sigy)
#             xn, xp = intround(midx-sigx), intround(midx+sigx)
#             maxI = image[yn:yp, xn:xp].mean()*2
#             minI = im.mean()
#             return midx, midy, sigx, sigy, maxI, minI
#         except:
#             if DEBUG:
#                 print(str(fact)+" failed:", im.mean(), fact*im.std())
#     return imx//2, imy//2, 5, 5, im.max(), im.min()

# def gauss1D_param(ydata):
#     for fact in [2, 1.5, 1, 0.5,0.25]:
#         try:
#             yd = ydata.copy()
#             idxs = yd < yd.mean() + fact*yd.std()
#             idxs = scipy.ndimage.binary_dilation(idxs)
#             yd[idxs] = 0
#             xs = np.argwhere(yd > 0)
#             if xs.size < 1: raise #check if list is empty
#             midx = xs.mean()
#             sigx = (xs.max() - xs.min())/2
#             xn, xp = intround(midx-sigx), intround(midx+sigx)
#             if yd[xn:xp].size < 1: raise #check if list is empty
#             maxI = yd[xn:xp].mean()*2
#             minI = ydata.mean()
#             if np.isnan(maxI) or sigx <0.5: raise
#             if DEBUG: 
#                 print("zprof ", str(fact)+" success:", ydata.mean(), fact*ydata.std())
#             return midx, 2*fact*sigx, maxI, minI
#         except:
#             if DEBUG:
#                 print("zprof ", str(fact)+" failed:", ydata.mean(), fact*ydata.std())
#     return int(len(ydata)/2), 5, max(ydata), min(ydata)

# def do_1D_gauss_fit(ydata, xdata=None):
#     if type(xdata) == type(None): xdata = np.arange(0, len(ydata))
#     midx, sigx, maxI, minI = gauss1D_param(ydata)
#     p0 = [xdata[intround(midx)], np.abs(xdata[1]-xdata[0])*sigx, maxI, minI]
#     popt, pcov = curve_fit(gaussian_1D, xdata, ydata, p0, maxfev = 8000)
#     xfine = np.linspace(xdata.min(), xdata.max(), len(xdata)*5)
#     return popt, xfine, xdata
