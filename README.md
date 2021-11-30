# PSF-Extractor

Created by Daan Boltje, Ernest van der Wee  
Made presentable by Ryan Lane  
Input from Yoram Vos

### Installation
* Create a new conda environment (assumes Anaconda or Miniconda is already installed) from the terminal
```
$ conda create -n psf -c conda-forge numpy scipy pandas matplotlib scikit-image jupyterlab trackpy
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
(psf) $ cd /path/to/PSF-Extractor/
(psf) $ jupyter lab
```

* Run `walkthrough.ipynb`
