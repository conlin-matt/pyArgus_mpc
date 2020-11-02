#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 14:18:32 2020

@author: matthewconlin
"""
import setuptools

setuptools.setup(
    name="pyArgus_mpc",
    version="0.1.0",
    url="https://github.com/conlin-matt/pyArgus_mpc",
    author="Matthew P. Conlin",
    author_email="conlinm@ufl.edu",
    description="Python implementations of some coastal imaging utilities.",
    packages=setuptools.find_packages(),
    install_requires=['matplotlib','numpy','scipy','scikit-image','scikit-learn'],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)