from setuptools import setup, find_packages
from os import path

description_file = path.join(path.abspath(path.dirname(__file__)), "DESCRIPTION.md")
with open(description_file, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = "ennemi",
    version = "1.0.0alpha1",
    description = "Easy-to-use Nearest Neighbor Estimation of Mutual Information",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://polsys.github.io/ennemi/",
    author = "Petri Laarne",
    author_email = "petri.laarne@helsinki.fi",
    license = "MIT",

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
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
        "Issues": "https://github.com/polsys/ennemi/issues"
    },

    packages = [ "ennemi" ],
    python_requires = "~=3.6",
    install_requires = [ "numpy~=1.13" ]
)