# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="CICEMOK",
    version="0.1.0",
    description="A pip package",
    license="MIT",
    author="oarcelus",
    packages=find_packages(),
    long_description=long_description,
    install_requires=required,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
    ]
)
