from setuptools import setup, find_packages

setup(
    name='microRicci',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'torch', 'matplotlib'
    ],
)
