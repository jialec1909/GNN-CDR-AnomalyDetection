#!/usr/bin/env python

from setuptools import find_packages, setup

with open("requirements.txt") as reqs_file:
    requirements = [req for req in reqs_file.read().splitlines() if not req.startswith(("#", "-"))]

setup(
    name="CDR",
    version="0.0.1",
    description="Repo for GNN on CDR",
    author="Jiale Chen",
    author_email="jiale.chen@stud.tu-darmstadt.de",
    python_requires=">=3.10",
    url="https://github.com/jialec1909/GNN-CDR-AnomalyDetection.git",
    install_requires=requirements,
    packages=find_packages(),
)