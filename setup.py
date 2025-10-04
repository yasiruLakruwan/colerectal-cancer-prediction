from setuptools import setup,find_packages
import os

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Colorectal_cancer_prediction_project",
    version="0.0.1",
    author="Yasiru",
    packages=find_packages(),
    install_requires = requirements
)