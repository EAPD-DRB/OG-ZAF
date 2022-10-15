try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md") as f:
    longdesc = f.read()

version = "0.0.0"

config = {
    "description": "South Africa Calibration for OG-Core",
    "long_description": longdesc,
    "url": "https://github.com/EAPD-DRB/OG-ZAF/",
    "download_url": "https://github.com/EAPD-DRB/OG-ZAF/",
    "version": version,
    "license": "CC0 1.0 Universal public domain dedication",
    "packages": ["ogzaf"],
    "include_package_data": True,
    "name": "ogzaf",
    "install_requires": [],
    "package_data": {"ogzaf": ["data/*"]},
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: CC0 1.0 Universal public domain dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "tests_require": ["pytest"],
}

setup(**config)
