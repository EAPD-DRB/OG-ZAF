import setuptools

with open("README.md", "r") as readme_file:
    longdesc = readme_file.read()

setuptools.setup(
    name="ogzaf",
    version="0.0.6",
    author="Marcelo LaFleur, Richard W. Evans, and Jason DeBacker",
    license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
    description="South Africa Calibration for OG-Core",
    long_description_content_type="text/markdown",
    long_description=longdesc,
    keywords="ZAF South Africa calibration of large scale overlapping generations model of fiscal policy",
    url="https://github.com/EAPD-DRB/OG-ZAF/",
    download_url="https://github.com/EAPD-DRB/OG-ZAF/",
    project_urls={
        "Issue Tracker": "https://github.com/EAPD-DRB/OG-ZAF/issues",
    },
    packages=["ogzaf"],
    package_data={
        "ogzaf": [
            "ogzaf_default_parameters.json",
            "ogzaf_default_parameters_multisector.json",
            "data/*",
        ]
    },
    include_packages=True,
    python_requires=">=3.7.7, <3.13",
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
        "wheel",
        "black",
        "linecheck",
        "ogcore",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Common Public License",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    tests_require=["pytest"],
)
