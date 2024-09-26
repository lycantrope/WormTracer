
from setuptools import find_packages, setup

setup(
    name='WormTracer',
    version="20.2rc2",
    description="WormTracer package",
    author='Koyo Kuze et al',
    packages=find_packages(),
    license='MIT',
    entry_points={ 'console_scripts': ['wormtracer = WormTracer.__main__:main' ] },
    install_requires=[
    "numpy",
    "opencv-python",
    "matplotlib",
    "Pillow",
    "scikit-image",
    "scipy",
    "pyyaml"
    ]
)
