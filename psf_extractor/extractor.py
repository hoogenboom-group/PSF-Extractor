from pathlib import Path
import multiprocessing
import psutil
import logging
from itertools import combinations

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from skimage import io, exposure, measure

from .util import natural_sort, bboxes_overlap, is_notebook
from .gauss import fit_gaussian_2D, fit_gaussian_1D

import tifffile

# Check for dask
try:
    import dask.array as da
    from dask.array.image import imread
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
           'save_stack', 
           'get_mip',
           'get_min_masses',
           'get_max_masses',
           'detect_overlapping_features',
           'detect_edge_features',
           'extract_psfs',
           'detect_outlier_psfs',
           'localize_psf',
           'localize_psfs',
           'filt_locations',
           'align_psfs',
           'crop_psf',
           'downsample_psf',
           'fit_features_in_stack',
           'eight_bit_as']


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
        stack = []
        for i, fp in tqdm(enumerate(file_pattern),
                          total=len(file_pattern)):
            logging.debug(f"Reading image file ({i+1}/{len(file_pattern)}) : {fp}")
            image = io.imread(fp, plugin='pil')
            stack.append(image)
            
        # Create 3D image stack (Length, Height, Width)
        stack = np.stack(stack, axis=0)

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
            stack = []
            for i, fp in tqdm(enumerate(filepaths),
                              total=len(filepaths)):
                logging.debug(f"Reading image file ({i+1}/{len(filepaths)}) : {fp}")
                image = io.imread(fp, plugin='pil')
                stack.append(image)
            # Create 3D image stack (Length, Height, Width)
            stack = np.stack(stack, axis=0)

        # Tiff stack or gif
        elif (Path(file_pattern).suffix == '.tif') or \
             (Path(file_pattern).suffix == '.tiff') or \
             (Path(file_pattern).suffix == '.gif'):
            logging.info("Creating stack from tiff stack")
            # Create 3D image stack (Length, Height, Width)
            stack = io.imread(file_pattern, plugin='pil')

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

    # Intensity rescaling (0 to 1 in float32)
    # Based on https://github.com/scikit-image/scikit-image/blob/main/skimage/exposure/exposure.py
    # Avoids additional float64 memmory allocation for data
    stack = stack.astype(np.float32)
    imin, imax = np.min(stack), np.max(stack)
    stack = np.clip(stack, imin, imax)
    stack -= imin
    stack /= (imax - imin)
    # Return stack
    logging.info(f"{stack.shape} image stack created succesfully.")
    return stack


def save_stack(psf, file_pattern,psx,psy,psz,usf,bin_num=None,bin_edges=None):
    """Save PSF to file, along with metadata of stack.
    
    Parameters
    ----------
    psf: array-like
        Image stack of shape (L, M, N)
        
    file_pattern : list or str
        Either a list of filenames or a string that is either:
        a) the individual filename of e.g. a tiff stack or
        b) a directory from which all images will be loaded into the stack
    bin_num: int
        bin number in case stack is binned (along x)
        
    Returns
    -------
    ...
    
    Notes
    -----
    Not tested on multiple files, but should in principle work.
    
    Escapes when PSF array has only zeros.
    """
    
    if np.sum(psf) == 0:
        if bin_num == None: 
            print('Empty PSF: only zeros...')
        else: 
            print('Empty PSF: only zeros... (Bin #'+str(bin_num)+')')
        return
    
    if isinstance(file_pattern, list):  # In case a list of files is provided
        if bin_num == None: 
            location = str(Path(file_pattern[0]).parent) + "/_output"
        else: 
            location = str(Path(file_pattern[0]).parent) + "/_output" + "/bin_"+str(bin_num)
        fp  = Path(location)
        fp.mkdir(exist_ok=True)
        #save psf to file
        tifffile.imwrite(location + "/psf_av.tif",eight_bit_as(psf,np.uint8),photometric='minisblack')
            
        #save meta data
        with open(location + '/parameters.txt','w') as f:
            f.write('stack parameters:\n')
            f.write('\n')
            f.write('X: '+str(psx/usf)+' nm\n')
            f.write('Y: '+str(psy/usf)+' nm\n')
            f.write('Z: '+str(psz/usf)+' nm\n')
    
    # If a single directory or multipage tiff is provided
    elif isinstance(file_pattern, str):
        if file_pattern[-1] == "/" or file_pattern[-1] == "\\": #directory!
            if bin_num == None: location = file_pattern + '_output'
            else: 
                location = file_pattern + '_output' + "/bin_"+str(bin_num)
            fp  = Path(location)
            fp.mkdir(exist_ok=True) #make output directory if not there
        else:    #file!
            if bin_num == None: location = str(Path(file_pattern).parent) + "/_output"
            else: 
                location = str(Path(file_pattern).parent) + "/_output" + "/bin_"+str(bin_num)
            fp  = Path(location)
            fp.mkdir(exist_ok=True) #make output directory if not there
        
        #save psf to file
        tifffile.imwrite(location + "/psf_av.tif",eight_bit_as(psf,np.uint8),photometric='minisblack')
        
        #save meta data
        with open(location + '/parameters.txt','w') as f:
            f.write('stack parameters:\n')
            f.write('\n')
            f.write('X: '+str(psx/usf)+' nm\n')
            f.write('Y: '+str(psy/usf)+' nm\n')
            f.write('Z: '+str(psz/usf)+' nm\n')
            if bin_num != None:
                bin_edges_upd = [0]
                bin_edges_upd.extend(bin_edges)
                bin_edges = [0].extend(bin_edges)
                f.write('\n')
                f.write('Bin center: ' + str(np.round(bin_edges_upd[bin_num-1]+bin_edges_upd[1]/2))+ ' pixels\n' )
                f.write('Bin center: ' + str(round(0.001*psx*(bin_edges_upd[bin_num-1]+bin_edges_upd[1]/2),2))+ ' microns' )
    
    if bin_num == None: print("Succesfully saved PSF and parameters to file.")
    else: print("Succesfully saved PSF and parameters of Bin #"+ str(bin_num) + " to file.")
    return

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
            dx ~ λ/(2NA)
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


def detect_overlapping_features(features, wx, wy=None):
    """Detects overlapping features from feature set.

    Parameters
    ----------
    features : `pd.DataFrame`
        Feature set returned from `trackpy.locate`
    wx, wy : scalar
        Dimensions of bounding boxes

    Returns
    -------
    overlapping : array-like
        Indices of overlapping features (to be discarded)

    Notes
    -----
    * Utilizes cell listing approach for huge speed increases over brute-force.
    """
    # Set wy if not provided
    wy = wx if wy is None else wy  # (assumes a square box)

    # Create a bounding box for each bead
    df_bboxes = features.loc[:, ['x', 'y']]
    df_bboxes['x_min'] = features['x'] - wx/2
    df_bboxes['y_min'] = features['y'] - wy/2
    df_bboxes['x_max'] = features['x'] + wx/2
    df_bboxes['y_max'] = features['y'] + wy/2

    # Keep track of overlapping features
    overlapping = []
    # Define cell parameters
    cw = 2*wx  # cell width
    # Alias for features
    X = features['x'].values
    Y = features['y'].values

    # Loop through a grid in x, y to create cells
    Nx = X.max() + cw
    Ny = Y.max() + cw
    for x in tqdm(np.arange(0, Nx, cw)):
        for y in np.arange(0, Ny, cw):
            # Create cell
            cell = [x-cw, y-cw, x+2*cw, y+2*cw]
            # Get features in cell
            in_cell = df_bboxes[((cell[0] < X) & (X < cell[2]) &\
                                 (cell[1] < Y) & (Y < cell[3]))]
            # Combinations
            pairs = list(combinations(in_cell.reset_index().values, 2))

            # Loop through pairs of bboxes
            for (bbox_i, bbox_j) in pairs:
                if bboxes_overlap(bbox_i[-4:], bbox_j[-4:]):
                    overlapping.append(bbox_i[0])
                    overlapping.append(bbox_j[0])

    # Deduplicate indices
    overlapping = np.unique(overlapping)
    return overlapping


def detect_edge_features(features, Dx, Dy, wx, wy=None):
    """Detects edge features from feature set.

    Parameters
    ----------
    features : `pd.DataFrame`
        Feature set returned from `trackpy.locate`
    Dx, Dy : scalar
        Dimensions of stack
    wx, wy : scalar
        Dimensions of bounding boxes

    Returns
    -------
    edges : array-like
        Indices of edge features (to be discarded)
    """
    # Set wy if not provided
    wy = wx if wy is None else wy  # (assumes a square box)

    # Create a bounding box for each bead
    df_bboxes = features.loc[:, ['x', 'y']]
    df_bboxes['x_min'] = features['x'] - wx/2
    df_bboxes['y_min'] = features['y'] - wy/2
    df_bboxes['x_max'] = features['x'] + wx/2
    df_bboxes['y_max'] = features['y'] + wy/2

    # Check boundaries
    edges = features.loc[(df_bboxes['x_min'] < 0) |\
                         (df_bboxes['y_min'] < 0) |\
                         (df_bboxes['x_max'] > Dx) |\
                         (df_bboxes['y_max'] > Dy)].index.values
    return edges


def extract_psfs(stack, features, shape):
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

    Returns
    -------
    psfs : list
        List of all the PSFs as numpy arrays
    features_upd : `pd.DataFrame`
        DataFrame of features with edge features removed

    Notes
    -----
    * A feature is considered to be an edge feature if the volume of the
      extracted PSF extends outside the image stack in z
    """
    # Unpack PSF shape
    wz, wy, wx = shape
    # Round up to nearest odd integer --> results in all extracted PSFs
    # having the same shape
    wz, wy, wx = np.ceil([wz, wy, wx]).astype(int) // 2 * 2 + 1
    if wz > stack.shape[0]: 
        logging.warning(f'Chosen PSF window z size ({wz} px) is larger '
                        f'than stack z size ({stack.shape[0]} px).')

    # Iterate through features
    psfs = []  # collect PSFs
    edge_features = []  # collect indices of edge features
    for i, row in features.iterrows():

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

        # Set z indices
        if stack.shape[0] < wz:  # image stack height < wz
            # Take full image stack in z
            z1, z2 = 0, stack.shape[0]
        else:
            # Find the max intensity along Z
            # Set width around center of bead to sum in-plane intensities
            pixel_width = 4 # Corresponds to the usual sampling rate (~8 pixels per bead) when recording a PSF.
            z_sum = stack[:, 
                          int(y-pixel_width):int(y+pixel_width), 
                          int(x-pixel_width):int(x+pixel_width)].sum(axis=(1, 2))
            max_index = np.argmax(z_sum)

            #set the volume around that point
            z1, z2 = (int(max_index - wz/2),
                      int(max_index + wz/2))

        # Determine if feature is along the edge of the image stack in z
        if (z1 <= 0) or (z2 > stack.shape[0]):
            edge_features.append(i)
        # Extract PSF
        else:
            psf = stack[z1:z2, y1:y2, x1:x2]
            psfs.append(psf)

    if len(psfs) == 0:
        logging.warning('\t All PSF windows outside of stack, all PSFs are filtered out. \
                         \n \t \t Decrease PSF window in Z (wz) or record larger Z-stack')

    # Remove edge features
    features_upd = features.drop(edge_features,axis=0)

    # Return psfs and updated feature set
    return psfs, features_upd


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
    # If no suspects exist
    if suspects_i.size == 0:
        outliers = np.array([])
    else:
        # Convert to indices of PSF pairs
        suspects_ij = np.array(ij)[suspects_i[:, 0]]

        # Determine frequency of out lying (?)
        i, counts = np.unique(suspects_ij, return_counts=True)
        outliers = i[counts > (pcc_min-0.1)*len(psfs)]
    if return_pccs:
        return outliers, pccs
    return outliers


def localize_psf(psf, integrate=False):
    """Localize a given PSF in the stack.

    Parameters
    ----------
    psf : array-like
        3D array of PSF subvolume
    integrate : bool
        Whether to integrate the PSF over x and y before doing 1D fit.
        Alternative is to take a slice in z at (x0, y0), the position
        found from the 2D fit.

    Returns
    -------
    x0, y0, z0 : scalars
        Position data from Gaussian fit
    sigma_x, sigma_y, sigma_z : scalars
        Standard deviations from Gaussian fit
    """
    # Take maximum intensity projection
    mip = np.max(psf, axis=0)

    # 2D Fit
    x0, y0, sigma_x, sigma_y, A, B = fit_gaussian_2D(mip)

    # 1D Fit
    # TODO: seems like slice is better but not totally convinced
    if integrate:
        # Integrate over x and y
        z_sum = psf.sum(axis=(1, 2))
        z0, sigma_z, A, B = fit_gaussian_1D(z_sum)
    else:
        # Slice in through x0, y0
        z_slice = psf[:,int(y0), int(x0)]
        z0, sigma_z, A, B = fit_gaussian_1D(z_slice)

    return (x0, y0, z0, sigma_x, sigma_y, sigma_z)


def localize_psfs(psfs, integrate=False):
    """Localize all PSFs in stack.

    Parameters
    ----------
    psfs : list or array-like
        List of PSFs
    integrate : bool
        Whether to integrate the PSF over x and y before doing 1D fit.

    Returns
    -------
    df : `pd.DataFrame`
        DataFrame of the fit parameters from each PSF
    """
    # Initialize DataFrame
    cols = ['x0', 'y0', 'z0', 'sigma_x', 'sigma_y', 'sigma_z']
    df = pd.DataFrame(columns=cols)
    # Loop through PSFs
    for i, psf in tqdm(enumerate(psfs), total=len(psfs)):
        try:
            # Localize each PSF and populate DataFrame with fit parameters
            df.loc[i, cols] = localize_psf(psf, integrate=integrate)
        # `curve_fit` failed
        except RuntimeError:
            logging.warning('Could not fit PSF, no location returned.')
            pass
    return df

def filt_locations(locations,features,psfs):
    """ Filter locations on their distance from center of PSF volume and width of fit (in X and Y)
        Using three sigma in the distribution
    
    Parameters
    ----------
    locations : pd dataframe
        fit results of fitting gaussians to the beads in PSF volume stack.
    features : pd dataframe
        list of features as obtained from trackpy
    psfs : 4d array
        array containing all extracted psfs [psf number, x, y, z]
    
    Returns
    ----------
    locations_new : pd dataframe
        filtered locations of the beads in PSF volume stack.
    features_new: pd dataframe
        filtered list of features as obtained from trackpy
    psfs: 4d array
        filtered array containing psfs [psf number, x, y, z]
    """
    
    #get distributions of x, y and sigmas
    x = locations.loc[:,'x0']
    y = locations.loc[:,'y0']
    sig_x = locations.loc[:,'sigma_x']
    sig_y = locations.loc[:,'sigma_y']
    
    # get three sigma lows and highs
    three_sigma_x = [x.mean() - 3 * x.std(), x.mean() + 3 * x.std()] 
    three_sigma_y = [y.mean() - 3 * y.std(), y.mean() + 3 * y.std()] 
    three_sigma_sig_x = [sig_x.mean() - 3 * sig_x.std(), sig_x.mean() + 3 * sig_x.std()] 
    three_sigma_sig_y = [sig_y.mean() - 3 * sig_y.std(), sig_y.mean() + 3 * sig_y.std()] 
    
    feat_index_list=(list(features.index.values) ) # this will always work in pandas
    #check if locations are within three sigma of x, y, sigma_x and sigma_y
    drop_list=[]
    for i in range(len(locations)):
        if (locations.loc[i,'x0'] < three_sigma_x[0] or locations.loc[i,'x0'] > three_sigma_x[1]
            or locations.loc[i,'y0'] < three_sigma_y[0] or locations.loc[i,'y0'] > three_sigma_y[1]
            or locations.loc[i,'sigma_x'] < three_sigma_sig_x[0] or locations.loc[i,'sigma_x'] > three_sigma_sig_x[1]
            or locations.loc[i,'sigma_y'] < three_sigma_sig_y[0] or locations.loc[i,'sigma_y'] > three_sigma_sig_y[1]):
            
            
            #make list of rows to drop
            drop_list.append(i)
            
    psfs = np.delete(psfs, drop_list, 0)
    
    if drop_list == []:
        features_new = features
    else:
        for j in drop_list:
            features_new=features.drop(feat_index_list[j], axis=0)
    
    locations_new=locations.drop(drop_list, axis=0)
    
    return locations_new, features_new,psfs

def align_psfs(psfs, locations, upsample_factor=2):
    """Upsample, align, and sum PSFs

    psfs : list or array-like
        List of PSFs
    locations : `pd.DataFrame`
        Localization data with z0, y0, and x0 positions
    upsample_factor : int
        Upsampling factor
        
    Returns
    -------
    psf_sum : array-like
        Aligned and summed together PSFs
    """
    # Alias
    usf = upsample_factor

    # Loop through PSFs
    psf_sum = 0  # dummy variable
    for i, psf in tqdm(enumerate(psfs), total=len(psfs)):

        # Upsample PSF
        psf_up = psfs[i].repeat(usf, axis=0)\
                        .repeat(usf, axis=1)\
                        .repeat(usf, axis=2)

        # From fit
        z0, y0, x0 = usf * locations.loc[i, ['z0', 'y0', 'x0']]
        # PSF center
        zc, yc, xc = (psf_up.shape[0]//2,
                      psf_up.shape[1]//2,
                      psf_up.shape[2]//2)

        # Multidimensional ~roll~ to align
        dz, dy, dx = int(zc-z0), int(yc-y0), int(xc-x0)
        psf_up_a = np.roll(psf_up, shift=(dz, dy, dx), axis=(0, 1, 2))

        # Sum PSFs
        psf_sum += psf_up_a

    return psf_sum


def crop_psf(psf, psx=None, psy=None, psz=None):
    """Crop an individual PSF to a cube."""
    # Get dimensions
    Nz, Ny, Nx = psf.shape

    # Cube of pixels
    if (psx is None) or (psy is None) or (psz is None):
        # Get smallest dimension
        N_min = np.min([Nz, Ny, Nx])
        # Crop psf to a cube defined by the smallest dimension
        z1, z2 = (Nz-N_min)//2, Nz - ((Nz-N_min)//2) - Nz % 2
        y1, y2 = (Ny-N_min)//2, Ny - ((Ny-N_min)//2) - Ny % 2
        x1, x2 = (Nx-N_min)//2, Nx - ((Nx-N_min)//2) - Nx % 2
        psf_cube = psf[z1:z2, y1:y2, x1:x2]

    # Cube of real units (um, nm)
    else:
        # Calculate real size of PSF
        dz, dy, dx = 1e-3*psz*Nz, 1e-3*psy*Ny, 1e-3*psx*Nx
        # Get smallet dimension
        d_min = np.min([dz, dy, dx])
        # Get center coords
        z0, y0, x0 = Nz//2, Ny//2, Nx//2
        # Crop psf to a cube defined by the smallest dimension
        z1, z2 = (z0 - int(d_min/2 / (1e-3*psz)),
                  z0 + int(d_min/2 / (1e-3*psz)))
        y1, y2 = (y0 - int(d_min/2 / (1e-3*psy)),
                  y0 + int(d_min/2 / (1e-3*psy)))
        x1, x2 = (x0 - int(d_min/2 / (1e-3*psx)),
                  x0 + int(d_min/2 / (1e-3*psx)))
        psf_cube = psf[z1:z2, y1:y2, x1:x2]

    return psf_cube

def downsample_psf(psf_sum, downsample_factor=2):
    psf = measure.block_reduce(psf_sum,
                               block_size=(downsample_factor,)*psf_sum.ndim,
                               func=np.mean,
                               cval=np.mean(psf_sum))
    return psf

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
    
def eight_bit_as(arr, dtype=np.float32):
    """Convert array to 8 bit integer array.
    
    Parameters
    ----------
    arr: array-like
        Array of shape (L, M, N)
        
    dtype : data type
        Data type of existing array.
        
    Returns
    -------
    arr.astype(dtype) : array-like
        Array formatted to 8 bit integer array
    
    Notes
    -----
    ...
    """
    
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr -= arr.min()
        arr *= 255./arr.max()
    else: 
        arr = arr.astype(np.float32)
    return arr.astype(dtype)
