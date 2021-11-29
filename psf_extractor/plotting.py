import numpy as np
import trackpy
from skimage import exposure

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .extractor import get_mip, get_min_masses, crop_psf
from .gauss import gaussian_1D, fit_gaussian_1D
from .util import get_Daans_special_cmap, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = ['plot_mip',
           'plot_mass_range',
           'plot_mass_range_interactive',
           'plot_psf',
           'plot_psfs',
           'plot_psf_localizations']


# No-brainer global variable
fire = get_Daans_special_cmap()


def plot_mip(mip, dx=None, dy=None, features=None):
    """Plot the maximum intensity projection

    Parameters
    ----------
    mip : array-like
        2D max intensity projection
    dx, dy : scalars
        Diameters in x and y
    features : `pd.DataFrame`
        DataFrame of features returned by `trackpy.locate`
    """
    # Round diameters up to nearest odd integer (as per `trackpy` instructions)
    if dx is not None:
        dx = int(np.ceil(dx)//2*2 + 1)
        dy = int(np.ceil(dx)//2*2 + 1) if dy is None else \
             int(np.ceil(dy)//2*2 + 1)
    # Locate features if not provided (but feature diameters are)
    if (dx is not None) and (features is None):
        # Locate features
        features = trackpy.locate(mip, diameter=[dy, dx]).reset_index(drop=True)

    # Enhance contrast in MIP (by taking the log)
    s = 1/mip[mip!=0].min()  # scaling factor (such that log(min) = 0
    mip_log = np.log(s*mip,
                     out=np.zeros_like(mip),
                     where=mip!=0)  # avoid /b0 error

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot MIP
    im = ax.imshow(mip_log, cmap=fire)
    # Plot features (if possible)
    if features is not None:
        ax.plot(features['x'], features['y'], ls='', color='#00ff00',
                marker='o', ms=15, mfc='none', mew=1)
        title = f'Features Detected: {len(features):.0f}'
    else:
        title = 'Maximum Intensity Projection'
    ax.set_title(title)
    # Colorbar
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)


def plot_mass_range(mip, dx, dy=None, masses=None, filtering='min',
                    **min_mass_kwargs):
    """Plot detected features from MIP for a range of masses

    Parameters
    ----------
    mip : array-like
        2D max intensity projection
    dx, dy : int
        Diameters in x and y
    masses : array-like
        1D list or array of masses used for filtering features from 
        `trackpy.locate`. Mass refers to the integrated brightness, which is 
        apparently "a crucial parameter for eliminating spurious features". 
        See trackpy documentation for details.
    filtering : str
        Whether to do min filtering or max filtering
        Default : 'min'
    min_mass_kwargs : dict
        Keyword arguments passed to `extractor.get_min_masses`
    """
    # Set candidate minimum masses if not provided
    if masses is None:
        masses = get_min_masses(mip, dx, **min_mass_kwargs)

    # Round diameters up to nearest odd integer (as per `trackpy` instructions)
    dx = int(np.ceil(dx)//2*2 + 1)
    dy = int(np.ceil(dx)//2*2 + 1) if dy is None else dy
    # Locate features
    features = trackpy.locate(mip, diameter=[dy, dx]).reset_index(drop=True)

    # Enhance contrast in MIP (by taking the log)
    s = 1/mip[mip!=0].min()  # scaling factor (such that log(min) = 0
    mip_log = np.log(s*mip,
                     out=np.zeros_like(mip),
                     where=mip!=0)  # avoid /b0 error

    # Set up figure
    ncols = 3
    nrows = int(np.ceil(len(masses) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 6*nrows))

    # Loop through candidate minimum masses
    for i, mass in tqdm(enumerate(masses),
                        total=len(masses)):
        # Filter based on (raw) mass
        if filtering == 'min':
            df = features.loc[features['raw_mass'] > mass]
        elif filtering == 'max':
            df = features.loc[features['raw_mass'] < mass]
        else:
            raise ValueError("`filtering` must be one of 'min' or 'max'.")

        # Plot max projection image
        ax = axes.flat[i]
        ax.imshow(mip_log, cmap=fire)
        # Plot detected features
        ax.plot(df['x'], df['y'], ls='', color='#00ff00',
                marker='o', ms=15, mfc='none', mew=1)
        title = f'Mass: {mass:.1f} | Features Detected: {len(df):.0f}'
        ax.set_title(title)


def plot_mass_range_interactive(mip, mass, features, filtering='min'):
    """Interactive plot to determine minimum or maxmimum mass threshold

    Parameters
    ----------
    mip : array-like
        2D max intensity projection
    mass : scalar
        Mass used for filtering features from `trackpy.locate`. Mass refers to 
        the integrated brightness, which is apparently "a crucial parameter for 
        eliminating spurious features". See trackpy documentation for details.
    features : `pd.DataFrame`
        DataFrame of features returned by `trackpy.locate`
    """
    # Enhance contrast in MIP (by taking the log)
    s = 1/mip[mip!=0].min()  # scaling factor (such that log(min) = 0
    mip_log = np.log(s*mip,
                     out=np.zeros_like(mip),
                     where=mip!=0)  # avoid /b0 error

    # Filter based on (raw) mass
    if filtering == 'min':
        df = features.loc[features['raw_mass'] > mass]
    elif filtering == 'max':
        df = features.loc[features['raw_mass'] < mass]
    else:
        raise ValueError("`filtering` must be one of 'min' or 'max'.")

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip_log, cmap=fire)  # plot MIP
    ax.plot(df['x'], df['y'], ls='', color='#00ff00',
            marker='o', ms=15, mfc='none', mew=1)
    title = f'Features Detected: {len(df):.0f}'
    ax.set_title(title)
    plt.show()


def plot_psf(psf, psx, psy, psz, crop=True):
    """"""

    # Create figure and axes
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(6, 6)
    ax_xy = fig.add_subplot(gs[:3,:3])
    ax_yz = fig.add_subplot(gs[:3,3:])
    ax_xz = fig.add_subplot(gs[3:,:3])
    ax_z = fig.add_subplot(gs[-3,3:])
    ax_y = fig.add_subplot(gs[-2,3:])
    ax_x = fig.add_subplot(gs[-1,3:])

    # Crippity crop crop
    if crop:
        psf = crop_psf(psf)

    # PSF dimensions
    Nz, Ny, Nx = psf.shape
    # PSF volume
    dz, dy, dx = psz*Nz/1e3, psy*Ny/1e3, psx*Nx/1e3
    # PSF center coords
    z0, y0, x0 = Nz//2, Ny//2, Nx//2

    # Plot 2D PSFs (slices)
    ax_xy.imshow(psf[z0,:,:], extent=[-dx/2, dx/2, -dy/2, dy/2], cmap=fire)
    ax_yz.imshow(psf[:,y0,:], extent=[-dz/2, dz/2, -dy/2, dy/2], cmap=fire)
    ax_xz.imshow(psf[:,:,x0], extent=[-dx/2, dx/2, -dz/2, dz/2], cmap=fire)
    # 1D PSFs (slices)
    prof_z = psf[:,y0,x0]
    prof_y = psf[z0,:,x0]
    prof_x = psf[z0,y0,:]
    # Do 1D PSF fits
    u = np.linspace(0, Nz, 10*Nz)
    popt_z = fit_gaussian_1D(prof_z)
    popt_y = fit_gaussian_1D(prof_y)
    popt_x = fit_gaussian_1D(prof_x)
    # Scale axes
    x_prof_z = np.linspace(-dz/2, dz/2, prof_z.size)
    x_prof_y = np.linspace(-dy/2, dy/2, prof_y.size)
    x_prof_x = np.linspace(-dx/2, dx/2, prof_x.size)
    x_uz = np.linspace(-dz/2, dz/2, u.size)
    x_uy = np.linspace(-dy/2, dy/2, u.size)
    x_ux = np.linspace(-dx/2, dx/2, u.size)

    # Plot 1D PSFs
    plot_kwargs = {'ms': 5, 'marker': 'o', 'ls': '', 'alpha': 0.75}
    ax_z.plot(x_prof_z, prof_z, c='C1', label='Z', **plot_kwargs)
    ax_y.plot(x_prof_y, prof_y, c='C0', label='Y', **plot_kwargs)
    ax_x.plot(x_prof_x, prof_x, c='C2', label='X', **plot_kwargs)

    # Plot 1D PSF fits
    ax_z.plot(x_uz, gaussian_1D(u, *popt_z), 'k-',
              label=f'{popt_z[1]:.2f}nm\nFWHM')
    ax_y.plot(x_uy, gaussian_1D(u, *popt_y), 'k-',
              label=f'{popt_y[1]:.2f}nm\nFWHM')
    ax_x.plot(x_ux, gaussian_1D(u, *popt_x), 'k-',
              label=f'{popt_x[1]:.2f}nm\nFWHM')

    # --- Aesthetics ---
    # XY projection
    ax_xy.text(0.02, 0.02, 'XY', fontsize=14, color='white', transform=ax_xy.transAxes)
    ax_xy.set_xlabel('X (μm)')
    ax_xy.set_ylabel('Y (μm)')
    ax_xy.xaxis.set_ticks_position('top')
    ax_xy.xaxis.set_label_position('top')
    # YZ projection
    ax_yz.text(0.02, 0.02, 'YZ', fontsize=14, color='white', transform=ax_yz.transAxes)
    ax_yz.set_xlabel('Z (μm)')
    ax_yz.set_ylabel('Y (μm)')
    ax_yz.xaxis.set_ticks_position('top')
    ax_yz.xaxis.set_label_position('top')
    ax_yz.yaxis.set_ticks_position('right')
    ax_yz.yaxis.set_label_position('right')
    # XZ projection
    ax_xz.text(0.02, 0.02, 'XZ', fontsize=14, color='white', transform=ax_xz.transAxes)
    ax_xz.set_xlabel('X (μm)')
    ax_xz.set_ylabel('Z (μm)')
    # Other stuff
    [ax.legend(loc='upper right') for ax in [ax_z, ax_y, ax_x]]
    [ax.grid(ls=':') for ax in [ax_z, ax_y, ax_x]]
    plt.subplots_adjust(hspace=0.5, wspace=0.5)


def plot_psfs(psfs):
    """Plot MIPs of extracted PSFs"""
    # Create figure
    ncols = 8
    nrows = int(np.ceil(len(psfs) / ncols))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(4*ncols, 4*nrows))
    # Loop through PSFs
    for i, psf in enumerate(psfs):
        ax = axes.flat[i]
        mip = np.max(psf, axis=0)
        ax.imshow(mip, cmap=fire)
        ax.set_title(f'PSF {i}')
    # Remove empty subplots
    [fig.delaxes(axes.flat[-i-1]) for i in range(ncols*nrows - len(psfs))]
    return fig


def plot_psf_localizations(df):
    """Plot PSF localizations

    Parameters
    ----------
    df : `pd.DataFrame`
        Localization data with columns "x0", "y0", "z0",
        "sigma_x", "sigma_y", "sigma_z" where (z0, y0, x0)
        is the center coordinate of the PSF within the stack
        and (sigma_z, sigma_y, sigma_x) is the fitted
        standard deviation of the PSF in each dimension

    Notes
    -----
    * Produces XY, YZ, and XZ projections of the fitted PSFs with the
      projections illustrated as matplotlib Ellipse patches
    """
    # Create figure
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

    # Loop through localization data
    for i, row in df.iterrows():
        # XY projection
        xy = row[['x0', 'y0']]
        w, h = row['sigma_x'], row['sigma_y']
        e = Ellipse(xy, w, h, alpha=0.1, color='C1')
        axes[0,0].add_patch(e)

        # YZ projection
        yz = row[['z0', 'y0']]
        w, h = row['sigma_z'], row['sigma_y']
        e = Ellipse(yz, w, h, alpha=0.1, color='C2')
        axes[0,1].add_patch(e)

        # XZ projection
        xz = row[['x0', 'z0']]
        w, h = row['sigma_x'], row['sigma_z']
        e = Ellipse(xz, w, h, alpha=0.1, color='C0')
        axes[1,0].add_patch(e)

    # PSF centers in x, y
    axes[0,0].plot(df['x0'], df['y0'], 'k+')  # XY
    axes[0,1].plot(df['z0'], df['y0'], 'k+')  # YZ
    axes[1,0].plot(df['x0'], df['z0'], 'k+')  # XZ

    # Aesthetics
    # ----------
    # XY projection
    axes[0,0].text(0.02, 0.02, 'XY', fontsize=14, transform=axes[0,0].transAxes)
    axes[0,0].set_xlabel('X [px]')
    axes[0,0].set_ylabel('Y [px]')
    axes[0,0].xaxis.set_ticks_position('top')
    axes[0,0].xaxis.set_label_position('top')
    # YZ projection
    axes[0,1].text(0.02, 0.02, 'YZ', fontsize=14, transform=axes[0,1].transAxes)
    axes[0,1].set_xlabel('Z [px]')
    axes[0,1].set_ylabel('Y [px]')
    axes[0,1].xaxis.set_ticks_position('top')
    axes[0,1].xaxis.set_label_position('top')
    axes[0,1].yaxis.set_ticks_position('right')
    axes[0,1].yaxis.set_label_position('right')
    # XZ projection
    axes[1,0].text(0.02, 0.02, 'XZ', fontsize=14, transform=axes[1,0].transAxes)
    axes[1,0].set_xlabel('X [px]')
    axes[1,0].set_ylabel('Z [px]')

    for ax in [axes[0,0], axes[1,0], axes[0,1]]:
        ax.set_aspect('equal')
        ax.grid(ls=':')
    fig.delaxes(axes[1,1])
