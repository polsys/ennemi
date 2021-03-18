# MIT License - Copyright Petri Laarne and contributors
# See the LICENSE.md file included in this source code package

from setuptools import setup, find_packages
from os import path

description_file = path.join(path.abspath(path.dirname(__file__)), "DESCRIPTION.md")
with open(description_file, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = "ennemi",
    version = "1.1.0",
    description = "Non-linear correlation detection with mutual information",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://polsys.github.io/ennemi/",
    author = "Petri Laarne",
    author_email = "petri.laarne@helsinki.fi",
    license = "MIT",

    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Typing :: Typed"
    ],
    keywords = "information-theory entropy mutual-information data-analysis scientific",

    project_urls = {
        "Documentation": "https://polsys.github.io/ennemi/",
        "Source": "https://github.com/polsys/ennemi/",
        "Issues": "https://github.com/polsys/ennemi/issues",
        "Zenodo": "https://doi.org/10.5281/zenodo.3834018"
    },

    packages = [ "ennemi" ],
    package_data = { "ennemi": ["py.typed"] },
    python_requires = "~=3.7",
    # At least pandas requires numpy 1.17.3+ (security fixes), we should too
    install_requires = [ "numpy>=1.17.5", "numpy<2.0", "scipy~=1.4" ],
    extras_require = {
        "dev": [ "pandas~=1.0", "pytest~=5.4", "mypy~=0.770" ]
    }
)