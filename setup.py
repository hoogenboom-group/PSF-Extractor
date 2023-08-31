from setuptools import setup

DISTNAME = 'PSF-Extractor'
DESCRIPTION = 'PSF-Extractor: Characterize the PSF of an optical setup'
MAINTAINER = 'Ryan Lane'
MAINTAINER_EMAIL = 'r.i.lane@tudelft.nl'
LICENSE = 'LICENSE'
README = 'README.md'
URL = 'https://github.com/hoogenboom-group/PSF-Extractor'
VERSION = '0.2'
PACKAGES = ['psf_extractor']
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'scikit-image',
    'trackpy',
    'tqdm',
    'dask',
    'joblib',
    'xarray',
    'mpl_interactions',
    'ipympl'
]

if __name__ == '__main__':

    setup(name=DISTNAME,
          version=VERSION,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          packages=PACKAGES,
          include_package_data=True,
          url=URL,
          license=LICENSE,
          description=DESCRIPTION,
          long_description=open(README).read(),
          install_requires=INSTALL_REQUIRES)
