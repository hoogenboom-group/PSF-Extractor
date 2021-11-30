from pathlib import Path
import multiprocessing
import psutil
import logging
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from skimage import io, exposure

from .util import natural_sort, bboxes_overlap, is_notebook
from .gauss import fit_gaussian_2D, fit_gaussian_1D


# Check for dask
try:
    import dask.array as da
    from dask_image.imread import imread
    _has_dask = True
except ImportError:
    logging.warn("Dask not installed. No support for large (> RAM) stacks.")
    _has_dask = False

# Determine whether in notebook environment (for tqdm aesthetics)
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# Get CPU info
N_CORES = multiprocessing.cpu_count()
MEM_TOT = psutil.virtual_memory().total / 1e9
MEM_FREE = psutil.virtual_memory().free / 1e9


__all__ = ['load_stack',
           'get_mip',
           'get_min_masses',
           'get_max_masses',
           'remove_overlapping_features',
           'extract_psfs',
           'detect_outlier_psfs',
           'align_psfs',
           'crop_psf',
           'fit_features_in_stack',
           'get_theta']

# TODO: LOGGING

def load_stack(file_pattern):
    """Loads image stack into dask array allowing manipulation
    of large datasets.

    Parameters
    ----------
    file_pattern : list or str
        Either a list of filenames or a string that is either
        a) the individual filename of e.g. a tiff stack or
        b) a directory from which all images will be loaded into the stack

    Returns
    -------
    stack : dask array-like
        Image stack as 32bit float with (0, 1) range in intensity

    Examples
    --------
    * `file_pattern` is a list
    >>> file_pattern = ['/path/to/data/image1.tif',
                        '/path/to/data/image2.tif',
                        '/path/to/data/image3.tif']
    >>> get_stack(file_pattern)

    * `file_pattern` is a directory
    >>> file_pattern = '/path/to/data/'
    >>> get_stack(file_pattern)

    * `file_pattern is a tiff stack
    >>> file_pattern = '/path/to/tiff/stack/multipage.tif'
    >>> get_stack(file_pattern)
    """

    # If a list of file names is provided
    if isinstance(file_pattern, list):
        logging.info("Creating stack from list of filenames.")
        images = []
        for i, fp in tqdm(enumerate(file_pattern),
                          total=len(file_pattern)):
            logging.debug(f"Reading image file ({i+1}/{len(file_pattern)}) : {fp}")
            image = io.imread(fp)
            images.append(image)
        # Create 3D image stack (Length, Height, Width)
        stack = np.stack(images, axis=0)

    # If a directory or individual filename
    elif isinstance(file_pattern, str):
        # Directory
        if Path(file_pattern).is_dir():
            logging.info("Creating stack from directory.")
            # Collect every png/tif/tiff image in directory
            filepaths = list(Path(file_pattern).glob('*.png')) + \
                        list(Path(file_pattern).glob('*.tif')) + \
                        list(Path(file_pattern).glob('*.tiff'))
            # Sort filepaths
            filepaths = natural_sort([fp.as_posix() for fp in filepaths])
            # Load images
            images = []
            for i, fp in tqdm(enumerate(filepaths),
                              total=len(filepaths)):
                logging.debug(f"Reading image file ({i+1}/{len(filepaths)}) : {fp}")
                image = io.imread(fp)
                images.append(image)
            # Create 3D image stack (Length, Height, Width)
            stack = np.stack(images, axis=0)

        # Tiff stack or gif
        elif (Path(file_pattern).suffix == '.tif') or \
             (Path(file_pattern).suffix == '.tiff') or \
             (Path(file_pattern).suffix == '.gif'):
            logging.info("Creating stack from tiff stack")
            # Create 3D image stack (Length, Height, Width)
            stack = io.imread(file_pattern)

        # ?
        else:
            if Path(file_pattern).exists():
                raise ValueError(f"Not sure what to do with `{file_pattern}`.")
            else:
                raise ValueError(f"`{file_pattern}` cannot be located or "
                                  "does not exist.")

    else:
        raise TypeError("Must provide a directory, list of filenames, or the "
                        "filename of an image stack as either a <list> or <str>, "
                        f"not {type(file_pattern)}.")

    # Return stack
    logging.info(f"{stack.shape} image stack created succesfully.")
    stack = exposure.rescale_intensity(stack, out_range=np.float32)
    return stack


def get_mip(stack, normalize=True, log=False, clip_pct=0, axis=0):
    """Compute the maximum intensity projection along the given axis.

    Parameters
    ----------
    stack : array-like
        3D image stack
    normalize : bool (optional)
        Whether to normalize the projection, also scales by 255
        Default : True
    log : bool (optional)
        Whether to take the natural log
        Default : False
    clip_pct : scalar (optional)
        % by which to clip the intensity
    axis : int (optional)
        Axis along which to compute the projection
        0 --> z, 1 --> y, 2 --> x
        Default : 0 (z)

    Returns
    -------
    mip : MxN array
        Maximum intensity projection image
    """
    # Calculate the maximum projection image of the image stack
    mip = np.max(stack, axis=axis)
    # Take natural log
    if log:
        # Scaling factor (such that log(min) = 0
        s = 1/mip[mip!=0].min()
        # Funky out + where arguments to avoid /b0 error
        mip = np.log(s*mip,
                     out=np.zeros_like(mip),
                     where=mip!=0)
    # Normalize (rescale) the maximum intensity projection
    if normalize or log or clip_pct:  # automatically rescale if taking
                                      # the log or if `clip_pct` provided
        p1, p2 = np.percentile(mip, (clip_pct, 100-clip_pct))
        mip = exposure.rescale_intensity(mip, in_range=(p1, p2), out_range=(0, 1))
    return mip


def get_min_masses(mip, dx, n=6, b=5):
    """Infer range of candidate minimum masses.

    Features returned by `trackpy.locate` are filtered by mass (essentially 
    a feature's total integrated brightness/intensity). It is important to 
    choose a reasonable lower bound for mass to filter out spurious bright 
    features (salt), smaller than the PSF, but it is difficult know what this 
    bound is a priori. So it is useful to sample a logarithmic range of 
    candidate lower bounds and choose a proper minimum mass based on visual 
    inspection.
    
    Parameters
    ----------
    mip : array-like
        2D maximum intensity projection
    dx : scalar
        Expected feature diameter
        A decent estimate is half the emissision wavelength divided by the NA
            dx ~ λ/(2×NA)
    n : scalar (optional)
        Number of candidate minimum masses to return
        Default : 6
    b : scalar (optional)
        Scaling factor to broaden or shrink the range of masses
        Default : 5

    Returns
    -------
    min_masses : array-like
        1D array of candidate minimum masses (length n)

    Examples
    --------
    >>> image = generate_image(nx=300, ny=300, N_features=20, seed=37)
    >>> get_min_masses(image, dx=9)
        array([ 12.21489226,  23.25292776, 44.26552752,
                84.26624581, 160.41377073, 305.37230648])
    """
    # Estimate peak intensity of a typical *single* PSF
    peak = np.percentile(mip, 99.9)
    # "Integrate" intensity over a typical PSF
    min_mass_0 = np.pi * (dx/2)**2 * peak
    # Set logarithmic range of candidate minimum masses
    min_masses = np.logspace(np.log10(min_mass_0/b),
                             np.log10(min_mass_0*b), n)
    return min_masses


def get_max_masses(min_mass, n=6, b=5):
    """Infer range of candidate maximum masses.

    Follows from `get_min_masses`, but for (surprise!) maximum mass filtering.
    Ranges from (min_mass, b*min_mass)

    Parameters
    ----------
    mip : array-like
        2D maximum intensity projection
    min_mass : scalar
        Minimum mass
    n : scalar (optional)
        Number of candidate maximum masses to return
        Default : 6
    b : scalar (optional)
        Scaling factor to broaden or shrink the range of masses
        Default : 5

    Returns
    -------
    max_masses : array-like
        1D array of candidate maximum masses (length n)
    """
    # Set logarithmic range of candidate maximum masses
    max_masses = np.logspace(np.log10(min_mass),
                             np.log10(min_mass*b), n)
    return max_masses


def remove_overlapping_features(features, wx, wy, return_indices=False):
    """Remove overlapping features from feature set.

    Parameters
    ----------
    features : `pd.DataFrame`
        Feature set returned from `trackpy.locate`
    wx, wy : scalar
        Dimensions of bounding boxes
    return_indices : bool
        Whether to return the indices of the overlapping features

    Returns
    -------
    features : `pd.DataFrame`
        Feature set with overlapping features removed
    """
    # TODO: figure out why this is so much slower than
    # previous blacklist function (which also checks
    # against the image borders) 
    
    # Create a bounding box for each bead
    df_bboxes = features.loc[:, ['x', 'y']]
    df_bboxes['x_min'] = features['x'] - wx
    df_bboxes['y_min'] = features['y'] - wy
    df_bboxes['x_max'] = features['x'] + wx
    df_bboxes['y_max'] = features['y'] + wy

    # Collect overlapping features
    overlapping_features = []
    # Iterate through every (unique) pair of bounding boxes
    ij = list(combinations(df_bboxes.index, 2))
    for i, j in tqdm(ij, total=len(ij)):

        # Create bounding boxes for each pair of features
        bbox_i = df_bboxes.loc[i, ['x_min', 'y_min', 'x_max', 'y_max']].values
        bbox_j = df_bboxes.loc[j, ['x_min', 'y_min', 'x_max', 'y_max']].values

        # Check for overlap
        if bboxes_overlap(bbox_i, bbox_j):
            overlapping_features.append(i)
            overlapping_features.append(j)

    overlapping = np.unique(overlapping_features)
    features = features.drop(index=overlapping)
    if return_indices:
        return features, overlapping
    return features


def extract_psfs(stack, features, shape, return_features=False):
    """Extract the PSF (aka subvolume) from each detected feature while 
    simultaneously filtering out edge features.

    Parameters
    ----------
    stack : array-like
        3D image stack
    features : `pd.DataFrame`
        DataFrame of detected features
    shape : array-like or 3-tuple
        The dimensions of the PSF to be extracted (wz, wy, wx)
    return_features : bool
        Whether to return updated feature set

    Returns
    -------
    psfs : list
        List of all the PSFs as numpy arrays
    features : `pd.DataFrame` (optional)
        DataFrame of features with edge features removed
        
    Notes
    -----
    * A feature is considered to be an edge feature if the volume of the
      extracted PSF extends outside the image stack in x or y
    """
    # Unpack PSF shape
    wz, wy, wx = shape
    # Round up to nearest odd integer --> results in all extracted PSFs
    # having the same shape
    wz, wy, wx = np.ceil([wz, wy, wx]).astype(int) // 2 * 2 + 1

    # Iterate through features
    psfs = []  # collect PSFs
    edge_features = []  # collect indices of edge features
    for i, row in features.iterrows():

        # Set z indices
        if stack.shape[0] < wz:  # image stack height < wz
            # Take full image stack in z
            z1, z2 = 0, stack.shape[0]
        else:
            # Place the subvolume at halfway in z
            z1, z2 = (int(stack.shape[0]/2 - wz/2),
                      int(stack.shape[0]/2 + wz/2))

        # Get x, y position of feature
        x, y = row[['x', 'y']]

        # Set y indices
        if stack.shape[1] < wy:  # image stack y width < wy
            # Take full image stack in y
            y1, y2 = 0, stack.shape[1]
        else:
            # Center the subvolume in y
            y1, y2 = (int(y - wy/2),
                      int(y + wy/2))

        # Set x indices
        if stack.shape[2] < wx:  # image stack x width < wx
            # Take full image stack x
            x1, x2 = 0, stack.shape[2]
        else:
            # Center the subvolume in x
            x1, x2 = (int(x - wx/2),
                      int(x + wx/2))

        # Determine if feature is along the edge of the image stack
        if (x1 < 0) or (y1 < 0) or (x2 > stack.shape[2]) or (y2 > stack.shape[1]):
            edge_features.append(i)
        # Extract PSF
        else:
            psf = stack[z1:z2, y1:y2, x1:x2]
            psfs.append(psf)

    # Donezo
    if not return_features:
        return psfs

    # Filter out and return edge features
    features = features.drop(edge_features).reset_index(drop=True)
    return psfs, features


def detect_outlier_psfs(psfs, pcc_min=0.9, return_pccs=False):
    """Detect outlier PSFs based on the Pearson correlation coefficient (PCC).

    Parameters
    ----------
    psfs : list
        List of PSFs
    pcc_min : scalar
        PCC threshold to determine suspicious (potential outlier) PSFs

    Returns
    -------
    outliers : list
        Indices of detected outlier PSFs
    """
    # Collect PCCs
    pccs = []
    # Iterate through every (unique) pair of PSFs
    ij = list(combinations(range(len(psfs)), 2))
    for i, j in tqdm(ij, total=len(ij)):

        # Get pairs of PSFs
        mip_i = np.max(psfs[i], axis=0)
        mip_j = np.max(psfs[j], axis=0)
        # Calculate PCC of maximum intensity projections
        pcc, _ = pearsonr(mip_i.ravel(),
                          mip_j.ravel())
        pccs.append(pcc)

    # Convert to array
    pccs = np.array(pccs)

    # Get indices of candidate outliers
    suspects_i = np.argwhere(pccs < pcc_min)
    # Convert to indices of PSF pairs
    suspects_ij = np.array(ij)[suspects_i[:, 0]]

    # Determine frequency of out lying (?)
    i, counts = np.unique(suspects_ij, return_counts=True)
    outliers = i[counts > 3*counts.mean()]

    if return_pccs:
        return outliers, pccs
    return outliers


def align_psfs(psfs, locations, upsample_factor=2):
    """Upsample, align, and sum PSFs.

    Parameters
    ----------
    psfs : list or array-like
        List of PSFs
    locations : `pd.DataFrame`
        Localization data with z0, y0, and x0 positions
    upsample_factor : scalar
        Factor by which to upsample the PSFs...

    Returns
    -------
    psf_sum : array-like
        Aligned and summed together PSFs
    """
    # Alias upsample factor
    usf = upsample_factor

    # Loop through and sum psfs
    psf_sum = 0
    for i, psf in tqdm(enumerate(psfs), total=len(psfs)):

        # Upsample PSFs
        psf_up = psf.repeat(usf, axis=0)\
                    .repeat(usf, axis=1)\
                    .repeat(usf, axis=2)

        # From fit
        z0, y0, x0 = usf * locations.loc[i, ['z0', 'y0', 'x0']]
        # PSF center
        zc, yc, xc = (psf_up.shape[0]//2,
                      psf_up.shape[1]//2,
                      psf_up.shape[2]//2)

        # Multidimensional ~roll~
        dz, dy, dx = int(zc-z0), int(yc-y0), int(xc-x0)
        psf_up_a = np.roll(psf_up, shift=(dz, dy, dx), axis=(0, 1, 2))

        # Sum PSFs
        psf_sum += psf_up_a

    return psf_sum


def crop_psf(psf):
    """Crop an individual PSF."""
    # Get dimensions
    Nz, Ny, Nx = psf.shape
    Nmin = np.min([Nz, Ny, Nx])

    # Crop total psf to a cube defined by the smallest dimension
    z1, z2 = (Nz-Nmin)//2, Nz - ((Nz-Nmin)//2) - Nz % 2
    y1, y2 = (Ny-Nmin)//2, Ny - ((Ny-Nmin)//2) - Ny % 2
    x1, x2 = (Nx-Nmin)//2, Nx - ((Nx-Nmin)//2) - Nx % 2
    psf_cube = psf[z1:z2, y1:y2, x1:x2]

    return psf_cube


def fit_features_in_stack(stack, features, width=None, theta=None):
    """Fit 2D gaussian to each slice in stack. XY positions
    defined 'x' and 'y' columns of features `pd.DataFrame'.

    Parameters
    ----------
    stack : array-like 
        Image stack of shape (L, M, N), L can be 0
    features : `pd.DataFrame`
        Feature set returned from `trackpy.locate`
    width : scalar
        Dimensions of bounding boxes
    theta : float or 2-valued tuple
        Angle bounds or estimate for elliptical 
        Gaussian fit

    Returns
    -------
    fit_features : `pd.DataFrame`
        DataFrame of resulting fit parameters for
        each feature defined in 'pd.DataFrame' features
        
    Notes
    -----
    ...
    """
    stack = np.array(stack)

    df_cols = ["x", "y", "sx", "sy", "A", "B"]
    if theta is not None: df_cols.insert(4, "t")
    if stack.ndim == 2: stack = [stack]

    # define cutout for each feature
    if width is None:
        width = 10 * features['size'].mean()
    df_bboxes = features.loc[:, ['x', 'y']]
    df_bboxes['x_min'] = features['x'] - width/2
    df_bboxes['y_min'] = features['y'] - width/2
    df_bboxes['x_max'] = features['x'] + width/2
    df_bboxes['y_max'] = features['y'] + width/2

    fit_results = []
    # iterate through stack
    for i, zslice in tqdm(enumerate(stack), total=len(stack)):
        fit_results.append([])
        logging.debug(f"Fitting slice ({i+1}/{len(stack)})")
        # for each zslice and each bead fit feature with 2D Gauss
        for j, row in df_bboxes.iterrows():
            x1, x2, y1, y2 = [int(p) for p in [row.x_min, row.x_max, 
                                               row.y_min, row.y_max]]
            feature_image = zslice[y1:y2, x1:x2]
            try:
                popt = fit_gaussian_2D(feature_image, theta=theta)
                fit_results[i].append(popt)
            except:
                fit_results[i].append(len(df_cols)*[np.nan])

    fr = np.array(fit_results)
    fit_features = pd.DataFrame()
    for i in range(fr.shape[1]):
        bead_df = (pd.DataFrame(fr[:, i, :], 
                                columns=df_cols)
                                .add_suffix(f"_{i}"))
        fit_features = pd.concat([fit_features, bead_df], axis=1)
    return fit_features
    
    
def get_theta(psf, fit_range=10):
    """Get theta from astigmatic PSF.

    Parameters
    ----------
    psf : array-like 
        Image stack of shape (L, M, N)

    Returns
    -------
    theta : float
        Astigmatic angle 
        
    Notes
    -----
    ...
    """
    z_sum = psf.sum(axis=(1, 2))
    z, sigma_z, A, B = fit_gaussian_1D(z_sum)
    
    z0, z1 = round(z - fit_range), round(z + fit_range)
    z = round(z)

    mip0, mip1 = get_mip(psf[z0:z]), get_mip(psf[z:z1])

    popt0 = fit_gaussian_2D(mip0, theta=(0,360))
    popt1 = fit_gaussian_2D(mip1, theta=popt0[4], epsilon=10)

    theta = np.mean([popt0[4], popt1[4]])

    return theta