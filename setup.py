import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='qramutils',
    version='0.1.0',
    author='Alessandro Scinawa Luongo',
    author_email='alessandro.luongo@atos.net',
    packages=['qramutils'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE',
    description='Library to get uneful statistics of a dataset for building a QRAM',
    #long_description=read('README.md'),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
)
