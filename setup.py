from setuptools import setup, find_packages

setup(
    name='CDR',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas~=1.5.3',
        'matplotlib~=3.7.1',
        'numpy~=1.25.0',
        'wandb~=0.15.12',
        'setuptools~=68.2.2'
    ],
    author='Jiale Chen',
    author_email='jialechen1909@gmail.com',
    description='GNN and transformer on CDR anomaly detection',
    url='https://github.com/jialec1909/GNN-CDR-AnomalyDetection',
)
