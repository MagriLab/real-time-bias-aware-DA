from setuptools import find_packages, setup

setup(
    name='real-time-DA',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Point root package directory at 'src'
    version='1.1',
    author='Andrea Nóvoa',
    author_email='a.novoa@imperial.ac.uk',
    description='Real-time Bias-Aware Data Assimilation',
)