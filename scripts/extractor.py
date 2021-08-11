import logging
from itertools import combinations
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
import numpy as np
import pandas as pd
import trackpy

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import pearsonr
from skimage.transform import resize

import psf_extractor as psfe
from psf_extractor.util import get_Daans_special_cmap


# TODO: make main function
# TODO: CLI interface


if __name__ == '__main__':

    # Set log level
    logging.getLogger().setLevel(logging.INFO)
    # Turn on interactive plotting
    plt.ion()

    # Request input
    # -------------
    # Request image stack directory or filenames
    fp = input("Filename or directory of input image stack: ")
    # Load image stack
    fp = './data/sample_zstack_png_sequence/'
    stack = psfe.load_stack(fp)

    # Request pixel sizes
    logging.info("Please input pixel size information.")
    psx = float(input("Pixel size in x (nm) [default = 64]: ") or "64")
    psy = float(input("Pixel size in y (nm) [default = 64]: ") or "64")
    psz = float(input("Pixel size in z (nm) [default = 125]: ") or "125")

    # Request expected feature diameters [nm]
    logging.info("Please input expected feature dimensions.")
    dx_nm = float(input("Expected feature diameter in x (nm) [default = 800]: ")\
                  or "800")
    dy_nm = float(input("Expected feature diameter in y (nm) [default = 800]: ")\
                  or "800")
    dz_nm = float(input("Expected feature diameter in z (nm) [default = 1500]: ")\
                  or "1500")

    # Convert expected feature diameters [nm --> px]
    dx = dx_nm / psx
    dy = dy_nm / psy
    dz = dz_nm / psz

    # Round diameters up to nearest odd integer (as per `trackpy` instructions)
    dx, dy, dz = np.ceil([dx, dy, dz]).astype(int) // 2*2+1

    # Maximum intensity projection
    # ----------------------------
    mip = psfe.get_mip(stack, axis=0, normalize=True)

    # Determine minimum mass
    # ----------------------
    # Set candidate minimum masses
    min_masses = [50, 100, 500, 1000, 2000, 5000]
    logging.info(f"Minimum masses: {min_masses}")
    logging.info("Finding features for each minimum mass...")
    # Make min mass figure
    psfe.plot_min_masses(stack, dx, dy, min_masses)
    logging.info("Generating plot...")
    # Request min mass
    min_mass = float(input("Choose minimum mass: "))
    plt.close()  # close plot

    # Locate features
    # ---------------
    logging.info("Detecting beads...")
    df_features = trackpy.locate(mip,
                                 diameter=[dy, dx],
                                 minmass=min_mass).reset_index(drop=True)
    n_beads = len(df_features)
    logging.info(f"{n_beads} beads detected.")

    # Filter out overlapping features
    # -------------------------------
    # Set bounding box dimensions [μm]
    dx_um = 5
    dy_um = dx_um
    # Convert bounding box dimensions [μm --> px]
    dx = dx_um / psx * 1e3
    dy = dy_um / psy * 1e3

    # Create a bounding box for each bead
    df_bboxes = df_features.loc[:, ['x', 'y']]
    df_bboxes['x_min'] = df_features['x'] - dx/2
    df_bboxes['y_min'] = df_features['y'] - dy/2
    df_bboxes['x_max'] = df_features['x'] + dx/2
    df_bboxes['y_max'] = df_features['y'] + dy/2

    # Collect overlapping features
    overlapping_features = []
    # Iterate through every (unique) pair of bounding boxes
    ij = list(combinations(df_bboxes.index, 2))
    logging.info("Checking beads for overlap...")
    for i, j in tqdm(ij, total=len(ij)):

        # Create bounding boxes for each pair of features
        bbox_i = df_bboxes.loc[i, ['x_min', 'y_min', 'x_max', 'y_max']].values
        bbox_j = df_bboxes.loc[j, ['x_min', 'y_min', 'x_max', 'y_max']].values

        # Check for overlap
        if psfe.bboxes_overlap(bbox_i, bbox_j):
            overlapping_features.append(i)
            overlapping_features.append(j)

    # Remaining features
    df_features = df_features.drop(index=overlapping_features)\
                             .reset_index(drop=True)
    logging.info(f"{n_beads - len(df_features)} beads removed for overlap.")
    n_beads = len(df_features)
    logging.info(f"{n_beads} beads remaining.")

    # Extract PSFs
    # ------------
    logging.info("Please choose dimensions of subvolumes for extracting "
                 "PSFs from the image stack.")
    # Set dimensions of subvolume [μm]
    wx_um = float(input("PSF dimension in x (μm) [default = 5]: ") or "5")
    wy_um = float(input("PSF dimension in y (μm) [default = 5]: ") or "5")
    wz_um = float(input("PSF dimension in z (μm) [default = 5]: ") or "5")
    # Convert dimensions of subvolume [μm --> px]
    wx = (wx_um / psx * 1e3)
    wy = (wy_um / psy * 1e3)
    wz = (wz_um / psz * 1e3)
    # Compile dimensions (z, y, x)
    shape_psf = [wz, wy, wx]
    # Extract PSFs
    logging.info("Extracting PSFs...")
    psfs, df_features_ = psfe.extract_psfs(stack, df_features, shape_psf)
    logging.info(f"Successfully extracted {len(psfs)} PSFs.")

    # Filter out "strange" features
    # -----------------------------
    logging.info("Filtering out PSFs based on Pearson correlation coefficient "
                 "(PCC)...")
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

    # Outlier criteria
    logging.info("Filtering out PSFs with a PCC outside a ±3 sigma range.")
    pcc_min = pccs.mean() - 3*pccs.std()
    pcc_max = pccs.mean() + 3*pccs.std()
    # Get indices of candidate outliers
    suspects_i = np.argwhere((pccs < pcc_min) |\
                             (pccs > pcc_max))
    # Convert to indices of PSF pairs
    suspects_ij = np.array(ij)[suspects_i[:, 0]]

    # Determine frequency of out lying (?)
    i, counts = np.unique(suspects_ij, return_counts=True)
    outliers = i[counts > 3*counts.mean()]

    # Remove outliers
    logging.info(f"Removing {len(outliers)} outliers.")
    df_features_ = df_features_.drop(outliers)

    # Re-extract PSFs based on updated feature set
    logging.info("Extracting PSFs based on updated feature set...")
    psfs, df_features = psfe.extract_psfs(stack, df_features_, shape_psf)
    logging.info(f"Successfully extracted {len(psfs)} PSFs.")

    # Least-squares fitting
    # ---------------------
    # Initialize localization DataFrame
    columns = ['x0', 'y0', 'z0', 'sigma_x', 'sigma_y', 'sigma_z']
    df_loc = pd.DataFrame(columns=columns)

    # Loop through PSFs
    for i, psf in enumerate(psfs):

        # --- 2D Fit ---
        # Take maximum intensity projection
        mip = np.max(psf, axis=0)
        x0, y0, sigma_x, sigma_y, A, B = psfe.fit_gaussian_2D(mip)

        # --- 1D Fit ---
        # Integrate over x and y
        z_sum = psf.sum(axis=(1, 2))
        z0, sigma_z, A, B = psfe.fit_gaussian_1D(z_sum)
        # Populate DataFrame
        df_loc.loc[i, columns] = [x0, y0, z0, sigma_x, sigma_y, sigma_z]

    # Upsample and align PSFs
    # -----------------------
    logging.info("Please choose an upsampling factor for aligning PSFs.")
    usf = float(input("Upsampling factor [default = 5]: ") or "5")
    logging.info("Aligning PSFs...")
    psf_sum = psfe.align_psfs(psfs, df_loc, upsample_factor=usf)

    # Generate plots
    # --------------
    figures = []
    logging.info("Generating PSF plot...")
    psf_fig = psfe.plot_psf(psf_sum, psx, psy, psz, crop=True)
    figures.append(psf_fig)

    # Export data and plots
    # ---------------------
    # Set export directory
    if isinstance(fp, list):
        # Set export directory to parent of first file in list
        export_dir = Path(fp[0]).parent / '_output/'
    elif Path(fp).is_dir():
        export_dir = Path(fp) / '_output/'
    else:  # is a multi-page tiff (hopefully)
        export_dir = Path(fp).parent / '_output'
    # Give option to select a different directory to export to
    logging.info("Please select a directory to store output.")
    export_dir = Path(input(f"Export directory [default = `{export_dir.as_posix()}`]: ") or \
                      export_dir.as_posix())
    # Create export directory
    logging.info(f"Exporting data to `{export_dir.absolute().as_posix()}`.")
    export_dir.mkdir(exist_ok=True)

    # --- Export stuff ---
    fp = export_dir / 'plots.pdf'
    with PdfPages(fp.as_posix()) as pdf:
        for fig in tqdm(figures):
            pdf.savefig(fig)

    logging.info("Finito, bro.")
