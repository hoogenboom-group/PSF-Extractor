# PSF-Extractor
A Python software package to characterize the Point Spread Function of an optical setup.

Created by Daan Boltje, Ernest van der Wee  
Made presentable by Ryan Lane  
Input from Yoram Vos

### Installation
* Create a new conda environment (assumes Anaconda or Miniconda is already installed) from the terminal
```
$ conda create -n psf -c conda-forge numpy scipy pandas matplotlib scikit-image jupyterlab trackpy xarray
```

* Activate environment
```
$ conda activate psf
```

* Install directly from github repository (assumes git is installed and in PATH)
```
(psf) $ pip install git+https://github.com/hoogenboom-group/PSF-Extractor.git
```


### To run in a Jupyter notebook
* (Optional) environment setup for enlightened folk
```
(psf) $ conda install -c conda-forge nodejs
(psf) $ pip install tqdm ipympl ipywidgets imagecodecs mpl_interactions
```

* Start jupyter lab session
```
(psf) $ cd /path/to/PSF-Extractor/
(psf) $ jupyter lab
```

* Run `walkthrough.ipynb`
