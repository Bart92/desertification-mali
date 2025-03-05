from setuptools import setup, find_packages

setup(
    name="desertification-mali",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "rasterio",
    ],
    python_requires=">=3.10",
)