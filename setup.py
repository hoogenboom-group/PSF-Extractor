from setuptools import setup, find_packages

DISTNAME = 'PSF-Extractor'
DESCRIPTION = 'PSF-Extractor: Some crazy mess Daan and Ernest made'
MAINTAINER = 'Ryan Lane'
MAINTAINER_EMAIL = 'r.i.lane@tudelft.nl'
LICENSE = 'LICENSE'
README = 'README.md'
URL = 'https://github.com/hoogenboom-group/PSF-Extractor'
VERSION = '0.1.dev'
PACKAGES = ['psf_extractor']
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'scikit-image',
    'trackpy'
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
