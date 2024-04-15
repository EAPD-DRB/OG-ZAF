"""This file contains the OG-ZAF package's metadata and dependencies."""

from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="ogzaf",
    version="0.0.1",
    author="Marcelo LaFleur, Richard W. Evans, and Jason DeBacker",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="South Africa Calibration for OG-Core",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Common Public License",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="South Africa calibration of large scale overlapping generations model of fiscal policy",
    url="https://github.com/EAPD-DRB/OG-ZAF/",
    download_url="https://github.com/EAPD-DRB/OG-ZAF/",
    project_urls={
        "Issue Tracker": "https://github.com/EAPD-DRB/OG-ZAF/issues",
    },
    packages=["ogzaf"],
    package_data={"ogzaf": ["ogusa_default_parameters.json", "data/*"]},
    include_packages=True,
    python_requires=">=3.7.7",
    install_requires=[
        "numpy",
        "psutil",
        "scipy>=1.7.1",
        "pandas>=1.2.5",
        "matplotlib",
        "dask>=2.30.0",
        "distributed>=2.30.1",
        "paramtools>=0.15.0",
        "requests",
        "pandas-datareader",
        "xlwt",
        "openpyxl>=3.1.2",
        "statsmodels",
        "linearmodels",
        "black",
        "linecheck",
        "ogcore",
    ],
    tests_require=["pytest"],
)
