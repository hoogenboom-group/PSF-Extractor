import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


__all__ = ['plot_psf',
           'plot_localization_data']


def plot_psf():
    pass


def plot_localization_data(df):
    """Plot localization data

    Parameters
    ----------
    df : `pd.DataFrame`
        Localization data with columns "x0", "y0", "z0",
        "sigma_x", "sigma_y", "sigma_z"
    
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
    axes[0,1].xaxis.set_ticks_position('top')
    axes[0,1].xaxis.set_label_position('top')
    axes[0,1].yaxis.set_ticks_position('right')
    # XZ projection
    axes[1,0].text(0.02, 0.02, 'XZ', fontsize=14, transform=axes[1,0].transAxes)
    axes[1,0].set_ylabel('Z [px]')

    for ax in [axes[0,0], axes[1,0], axes[0,1]]:
        ax.set_aspect('equal')
        ax.grid(ls=':')
    fig.delaxes(axes[1,1])
