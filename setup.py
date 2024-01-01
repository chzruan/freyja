from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
import subprocess
import os
from os.path import abspath, dirname, join
from glob import glob

this_dir = abspath(dirname(__file__))



setup(
    name='freyja',
    author="Cheng-Zong Ruan",
    author_email="chzruan@gmail.com",
    version='0.1dev',
    packages=['freyja'],
    license='MIT License',
    long_description=open('README.md').read(),
    #install_requires=requirements,
    zip_safe=False
)

