# PSF-Extractor
This crazy thing Daan and Ernest wrote

Created by Daan Boltje, Ernest van der Wee, and Yoram Vos  
Made presentable by Ryan Lane

### Installation
* Create a new conda environment (assumes Anaconda or Miniconda is already installed)
```
$ conda create -n psf -c conda-forge numpy scipy pandas matplotlib scikit-image jupyterlab trackpy pims
```

* Activate environment
```
$ conda activate psf
```

* Install directly from github repository
```
(psf) $ pip install git+git://github.com/hoogenboom-group/PSF-Extractor.git
```


### To run in a Jupyter notebook
* (Optional) environment setup for enlightened folk
```
(psf) $ conda install -c conda-forge nodejs=15
(psf) $ pip install tqdm ipympl ipywidgets imagecodecs
(psf) $ jupyter labextension install @jupyter-widgets/jupyterlab-manager
(psf) $ jupyter labextension install jupyter-matplotlib
(psf) $ jupyter nbextension enable --py widgetsnbextension
```

* Start jupyter lab session
```
(psf) $ jupyter lab
```
