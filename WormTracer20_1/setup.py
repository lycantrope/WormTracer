from setuptools import setup, find_packages

setup(
    name='WormTracer',
    version="20.1",
    description="WormTracer package",
    author='Koyo Kuze et al',
    packages=find_packages(),
    license='MIT',
    entry_points={ 'console_scripts': ['wormtracer = WormTracer.__main__:main' ] },
    install_requires=[
    "torch",
    "numpy",
    "opencv-python",
    "matplotlib",
    "Pillow",
    "scikit-image",
    "scipy",
    "pyyaml"
    ]
)
