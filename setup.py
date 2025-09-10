from setuptools import setup, find_packages

setup(
    name="senior_xgboost",
    version="1.0.0",
    author="Your Name",
    description="An advanced, from-scratch implementation of XGBoost for educational purposes.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
