import logging

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from .extractor import fit_features_in_stack
from .util import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def pfit_features_in_stack(stack, features, width=None, theta=None,
                                   num_cores=4):
    """Parallelized version of `fit_features_in_stack' 

    Parameters
    ----------
    stack : array-like or list of filenames
        Image stack of shape (L, M, N), L can be 0
    features : `pd.DataFrame`
        Feature set returned from `trackpy.locate`
    width : scalar
        Dimensions of bounding boxes
    theta : float or 2-valued tuple
        Angle bounds or estimate for elliptical 
        Gaussian fit
    num_cores: int
        Number of cores to use

    Returns
    -------
    fit_features : `pd.DataFrame`
        DataFrame of resulting fit parameters for
        each feature defined in 'pd.DataFrame' features
        
    Notes
    -----
    ...
    """
    fit_features = Parallel(n_jobs=num_cores) \
                           (delayed(fit_features_in_stack) \
                           (stack[i], features, width, theta) \
                           for i in tqdm(range(len(stack)), leave=True))
    return pd.concat(fit_features, ignore_index=True)
    