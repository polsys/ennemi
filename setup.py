from setuptools import setup, find_packages

setup(
    name = "ennemi",
    version = "1.0.0a1",
    description = "Easy-to-use Nearest Neighbor Estimation of Mutual Information",
    # TODO: long_description, urls, author, classifiers, keywords, ...
    packages = [ "ennemi" ],
    python_requires = "~=3.6",
    install_requires = [ "numpy~=1.13" ]
)