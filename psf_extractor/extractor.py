from pathlib import Path
import logging

import numpy as np
from skimage import img_as_float
from skimage import io


def get_stack(file_pattern):
    """Loads image stack

    Parameters
    ----------
    file_pattern : list or str
        Either a list of filenames or a string that is either
        1) the individual filename of e.g. a tiff stack or
        2) a directory from which all images will be loaded into the stack

    Returns
    -------
    stack : array-like
        Image stack as 64bit float with range of intensity values (0, 1)

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
        for i, fp in enumerate(file_pattern):
            logging.info(f"Reading image file ({i+1}/{len(file_pattern)}) : {fp}")
            image = img_as_float(io.imread(fp))
            images.append(image)
        # Create 3D image stack (Length, Height, Width)
        stack = np.stack(images, axis=0)

    # If a directory or individual filename
    elif isinstance(file_pattern, str):
        # Directory
        if Path(file_pattern).is_dir():
            logging.info("Creating stack from directory.")
            # Load every png/tif/tiff image in directory
            filepaths = list(Path(file_pattern).glob('*.png')) + \
                        list(Path(file_pattern).glob('*.tif')) + \
                        list(Path(file_pattern).glob('*.tiff'))
            images = []
            for i, fp in enumerate(filepaths):
                logging.info(f"Reading image file ({i+1}/{len(filepaths)}) : {fp.as_posix()}")
                image = img_as_float(io.imread(fp.as_posix()))
                images.append(image)
            # Create 3D image stack (Length, Height, Width)
            stack = np.stack(images, axis=0)

        # Tiff stack or gif
        elif (Path(file_pattern).suffix == '.tif') or \
             (Path(file_pattern).suffix == '.tiff') or \
             (Path(file_pattern).suffix == '.gif'):
            logging.info("Creating stack from tiff stack")
            # Create 3D image stack (Length, Height, Width)
            stack = img_as_float(io.imread(file_pattern))

        # ?
        else:
            raise ValueError(f"Not sure what to do with {file_pattern}.")

    else:
        raise TypeError("Must provide a directory, list of filenames, or the "
                        "filename of an image stack as either a <list> or <str>, "
                        f"not {type(file_pattern)}.")

    return stack
