from setuptools import find_packages, setup

setup(
    name="WormTracer",
    version="20.6.2",
    description="WormTracer package",
    author="Koyo Kuze et al",
    packages=find_packages(),
    entry_points={"console_scripts": ["wormtracer = WormTracer.__main__:main"]},
    install_requires=[
        "numpy",
        "opencv-python-headless",
        "matplotlib",
        "Pillow",
        "scikit-image",
        "scipy",
        "pyyaml",
        "roifile[all]>=2024.9.15",
    ],
)
