from setuptools import setup, find_packages

setup(
    name="cryosiam-vis",
    version="1.0",
    author="Frosina Stojanovska",
    author_email="stojanovska.frose@gmail.com",
    description="CryoSiam-Vis: Visualization Tool for CryoSiam Predictions",
    long_description_content_type="text/markdown",
    url="https://github.com/frosinastojanovska/cryosiam",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "cryosiam_vis = cryosiam_vis.cli:main"
        ]
    },
    install_requires=[
        "dash==2.14.2",
        "pandas==2.2.3",
        "PyYAML==6.0.2",
        "dash_bootstrap_components==1.7.1",
        "mrcfile==1.5.3",
        "h5py==3.11.0",
        "scipy==1.9.1",
        "dash-slicer==0.3.1",
        "napari==0.4.17",
        "starfile==0.5.12"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
