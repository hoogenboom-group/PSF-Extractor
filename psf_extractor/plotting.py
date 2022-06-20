import numpy as np
import trackpy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .extractor import get_mip, get_min_masses
from .gauss import gaussian_1D, fit_gaussian_1D, guess_gaussian_1D_params
from .util import get_Daans_special_cmap, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = ['plot_mip',
           'plot_mass_range',
           'plot_mass_range_interactive',
           'plot_overlapping_features',
           'plot_psf',
           'plot_psfs',
           'plot_psf_localizations',
           'plot_pcc_distribution']


# No-brainer global variable
fire = get_Daans_special_cmap()


def plot_mip(mip, dx=None, dy=None, features=None):
    """Plot the maximum intensity projection.

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
    """Plot detected features from MIP for a range of masses.

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
    """Interactive plot to determine minimum or maxmimum mass threshold.

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


def plot_overlapping_features(mip, features, overlapping, wx, wy=None):
    """Plot detected features from MIP for a range of masses.

    Parameters
    ----------
    mip : array-like
        2D max intensity projection
    features : `pd.DataFrame`
        DataFrame of detected features
    overlapping : array-like
        1D list or array of indices corresponding to overlapping features
    wx, wy : scalar
        Width and height of bounding box
    """
    # Set height if not provided
    if wy is None:
        wy = wx

    # Enhance contrast in MIP (by taking the log)
    s = 1/mip[mip!=0].min()  # scaling factor (such that log(min) = 0
    mip_log = np.log(s*mip,
                     out=np.zeros_like(mip),
                     where=mip!=0)  # avoid /b0 error

    # Set up figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot MIP
    ax.imshow(mip_log, cmap=fire)
    # Plot bboxes of overlapping and nonoverlapping features
    count = 0
    for i, feature in features.iterrows():
        if i in overlapping:
            color = '#ff0000'  # red
        else:
            color = '#00ff00'  # green
            count += 1
        p = Rectangle((feature['x']-wx/2, feature['y']-wy/2),
                      wx, wy, facecolor='none', lw=1, edgecolor=color)
        ax.add_patch(p)
    title = f'Non-overlapping features: {count:.0f}/{len(features):.0f}'
    ax.set_title(title)


def plot_psf(psf, psx, psy, psz):
    """Fancy PSF plot."""

    # Create figure and axes
    fig = plt.figure(figsize=(11, 11))
    gs = fig.add_gridspec(9, 9)
    ax_xy = fig.add_subplot(gs[:3,:3])
    ax_yz = fig.add_subplot(gs[:3,3:])
    ax_xz = fig.add_subplot(gs[3:,:3])
    ax_z = fig.add_subplot(gs[3:5,3:])
    ax_y = fig.add_subplot(gs[5:7,3:])
    ax_x = fig.add_subplot(gs[7:9,3:])

    # PSF dimensions
    Nz, Ny, Nx = psf.shape
    # PSF volume [μm]
    wz, wy, wx = 1e-3*psz*Nz, 1e-3*psy*Ny, 1e-3*psx*Nx
    # PSF center coords
    z0, y0, x0 = Nz//2, Ny//2, Nx//2

    # --- 2D Plots ---
    # Determine cropping margin
    crop_yz = int((wz - 2*wy) / (2*psz*1e-3)) if wz > 2*wy else None
    crop_xz = int((wz - 2*wx) / (2*psz*1e-3)) if wz > 2*wx else None
    # Crop 2D PSFs to 2:1 aspect ratio
    psf_z0 = psf[z0, :, :]
    psf_y0 = psf[crop_yz:-crop_yz, y0, :] if wz > 2*wy else psf[:, y0, :]
    psf_x0 = psf[crop_xz:-crop_xz, :, x0] if wz > 2*wx else psf[:, :, x0]
    # Update extent (after cropping)
    wz_cropped = psf_y0.shape[0] * 1e-3*psz
    # Plot 2D PSFs
    ax_xy.imshow(psf_z0, cmap=fire, interpolation='none',
                 extent=[-wx/2, wx/2, -wy/2, wy/2])
    ax_yz.imshow(psf_y0.T, cmap=fire, interpolation='none',
                 extent=[-wz_cropped/2, wz_cropped/2, -wy/2, wy/2])
    ax_xz.imshow(psf_x0, cmap=fire, interpolation='none',
                 extent=[-wx/2, wx/2, -wz_cropped/2, wz_cropped/2])

    # --- 1D Plots ---
    # 1D PSFs (slices)
    prof_z = psf[:, y0, x0]
    prof_y = psf[z0, :, x0]
    prof_x = psf[z0, y0, :]
    # 1D Axes
    z = np.linspace(-wz/2, wz/2, prof_z.size)
    y = np.linspace(-wy/2, wy/2, prof_y.size)
    x = np.linspace(-wx/2, wx/2, prof_x.size)
    # Do 1D PSF fits
    popt_z = fit_gaussian_1D(prof_z, z, p0=guess_gaussian_1D_params(prof_z, z))
    popt_y = fit_gaussian_1D(prof_y, y, p0=guess_gaussian_1D_params(prof_y, y))
    popt_x = fit_gaussian_1D(prof_x, x, p0=guess_gaussian_1D_params(prof_x, x))
    # Plot 1D PSFs
    plot_kwargs = {'ms': 5, 'marker': 'o', 'ls': '', 'alpha': 0.75}
    ax_z.plot(z, prof_z, c='C1', label='Z', **plot_kwargs)
    ax_y.plot(y, prof_y, c='C0', label='Y', **plot_kwargs)
    ax_x.plot(x, prof_x, c='C2', label='X', **plot_kwargs)
    # Plot 1D PSF fits
    ax_z.plot(z, gaussian_1D(z, *popt_z), 'k-')
    ax_y.plot(y, gaussian_1D(y, *popt_y), 'k-')
    ax_x.plot(x, gaussian_1D(x, *popt_x), 'k-')

    # --- FWHM arrows ---
    # Z
    x0 = popt_z[0]
    y0 = 0.65*(popt_z[2] + popt_z[3])
    fwhm = np.abs(2.355 * popt_z[1])
    ax_z.annotate('', xy=(x0-fwhm/2-0.5, y0), xytext=(x0-fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_z.annotate('', xy=(x0+fwhm/2+0.5, y0), xytext=(x0+fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_z.text(x0,popt_z[3], f'{1e3*fwhm:.0f}nm', ha='center')
    # Y
    x0 = popt_y[0]
    y0 = 0.65*(popt_y[2] + popt_y[3])
    fwhm = np.abs(2.355 * popt_y[1])
    ax_y.annotate('', xy=(x0-fwhm/2-0.5, y0), xytext=(x0-fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_y.annotate('', xy=(x0+fwhm/2+0.5, y0), xytext=(x0+fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_y.text(x0, popt_z[3], f'{1e3*fwhm:.0f}nm', ha='center')
    # X
    x0 = popt_x[0]
    y0 = 0.65*(popt_x[2] + popt_x[3])
    fwhm = np.abs(2.355 * popt_x[1])
    ax_x.annotate('', xy=(x0-fwhm/2-0.5, y0), xytext=(x0-fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_x.annotate('', xy=(x0+fwhm/2+0.5, y0), xytext=(x0+fwhm/2, y0), arrowprops={'arrowstyle': '<|-'})
    ax_x.text(x0, popt_z[3], f'{1e3*fwhm:.0f}nm', ha='center')

    # --- Aesthetics ---
    # XY projection
    ax_xy.text(0.02, 0.02, 'XY', color='white', fontsize=14, transform=ax_xy.transAxes)
    ax_xy.set_xlabel('X [μm]')
    ax_xy.set_ylabel('Y [μm]')
    ax_xy.xaxis.set_ticks_position('top')
    ax_xy.xaxis.set_label_position('top')
    # YZ projection
    ax_yz.text(0.02, 0.02, 'YZ', color='white', fontsize=14, transform=ax_yz.transAxes)
    ax_yz.set_xlabel('Z [μm]')
    ax_yz.set_ylabel('Y [μm]')
    ax_yz.xaxis.set_ticks_position('top')
    ax_yz.xaxis.set_label_position('top')
    ax_yz.yaxis.set_ticks_position('right')
    ax_yz.yaxis.set_label_position('right')
    # XZ projection
    ax_xz.text(0.02, 0.02, 'XZ', color='white', fontsize=14, transform=ax_xz.transAxes)
    ax_xz.set_xlabel('X [μm]')
    ax_xz.set_ylabel('Z [μm]')
    # 1D Axes
    ax_x.set_xlabel('Distance [μm]')
    [ax.set_xlim(-wy*1.1, wy*1.1) for ax in [ax_z, ax_y, ax_x]]
    # Miscellaneous
    [ax.legend(loc='upper right') for ax in [ax_z, ax_y, ax_x]]
    [ax.grid(ls=':') for ax in [ax_z, ax_y, ax_x]]
    plt.subplots_adjust(hspace=0.5, wspace=0.5)


def plot_psfs(psfs, psx=None, psy=None):
    """Plot MIPs of extracted PSFs."""
    # Switch to physical units if pixel sizes are provided
    if (psx is not None) and (psy is not None):
        # PSFs should all have same shape
        Nz, Ny, Nx = psfs[0].shape
        dy, dx = 1e-3*psy*Ny, 1e-3*psx*Nx
        extent = [-dx/2, dx/2, -dy/2, dy/2]
    else:
        extent = None

    # Create figure
    ncols = 8
    nrows = int(np.ceil(len(psfs) / ncols))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize=(4*ncols, 4*nrows))
    # Loop through PSFs
    for i, psf in tqdm(enumerate(psfs), total=len(psfs)):
        ax = axes.flat[i]
        mip = np.max(psf, axis=0)
        ax.imshow(mip, cmap=fire, interpolation='none',
                  extent=extent)
        ax.set_title(f'PSF {i}')
    # Remove empty subplots
    [fig.delaxes(axes.flat[-i-1]) for i in range(ncols*nrows - len(psfs))]


def plot_psf_localizations(df_input,psx,psy,psz,shape):
    """Plot PSF localizations.

    Parameters
    ----------
    df : `pd.DataFrame`
        Localization data with columns "x0", "y0", "z0",
        "sigma_x", "sigma_y", "sigma_z" where (z0, y0, x0)
        is the center coordinate of the PSF within the stack
        and (sigma_z, sigma_y, sigma_x) is the fitted
        standard deviation of the PSF in each dimension
    psx, psy, psz : float
        pixel size in x, y and z (in nm)
    shape : 3 - tuple
        The dimensions of the PSF to be extracted (wz, wy, wx)
    Notes
    -----
    * Produces XY, YZ, and XZ projections of the fitted PSFs with the
      projections illustrated as matplotlib Ellipse patches
    """
    
    # Alias
    df = df_input.copy()
    
    # Transform positions to mean of all positions
    x_center, y_center, z_center = np.mean(df['x0']),np.mean(df['y0']),np.mean(df['z0'])
    df['x0']=df['x0']-x_center
    df['y0']=df['y0']-y_center
    df['z0']=df['z0']-z_center
    
    # Convert to nanometers
    df ['x0'] = df ['x0'] * psx
    df ['y0'] = df ['y0'] * psy
    df ['z0'] = df ['z0'] * psz
    df ['sigma_x'] = df ['sigma_x'] * psx
    df ['sigma_y'] = df ['sigma_y'] * psy
    df ['sigma_z'] = df ['sigma_z'] * psz
    
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

    # --- Aesthetics ---
    # XY projection
    axes[0,0].text(0.02, 0.02, 'XY', fontsize=14, transform=axes[0,0].transAxes)
    axes[0,0].set_xlabel('X [nm]')
    axes[0,0].set_ylabel('Y [nm]')
    axes[0,0].xaxis.set_ticks_position('top')
    axes[0,0].xaxis.set_label_position('top')
    
    # YZ projection
    axes[0,1].text(0.02, 0.02, 'YZ', fontsize=14, transform=axes[0,1].transAxes)
    axes[0,1].set_xlabel('Z [nm]')
    axes[0,1].set_ylabel('Y [nm]')
    axes[0,1].xaxis.set_ticks_position('top')
    axes[0,1].xaxis.set_label_position('top')
    axes[0,1].yaxis.set_ticks_position('right')
    axes[0,1].yaxis.set_label_position('right')
    # XZ projection
    axes[1,0].text(0.02, 0.02, 'XZ', fontsize=14, transform=axes[1,0].transAxes)
    axes[1,0].set_xlabel('X [nm]')
    axes[1,0].set_ylabel('Z [nm]')

    for ax in [axes[0,0], axes[1,0], axes[0,1]]:
        ax.set_aspect('equal')
        ax.grid(ls=':')
    fig.delaxes(axes[1,1])


def plot_pcc_distribution(pccs, pcc_min=0.9, bins=None):
    """Plot distribution of PCCs of the extracted PSFs."""
    # Get distribution
    bins = pccs.size if bins is None else bins
    hist, bins = np.histogram(pccs, bins=bins)
    bins = (bins[1:] + bins[:-1]) / 2  # center bins

    # Plot distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bins, hist)
    ax.fill_between(bins, 0, hist, alpha=0.3)
    # Plot outlier range
    ax.axvline(pcc_min, ymax=1, color='k', ls='--')
    # Aesthetics
    ax.set_xlabel('PCC [AU]')
    ax.set_ylabel('Freq')
    ax.set_title('Distribution of PCCs')
    ax.grid(ls=':')

