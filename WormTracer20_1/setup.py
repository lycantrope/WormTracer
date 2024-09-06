from setuptools import setup, find_packages

setup(
    name='WormTracer',
    version="20.1.dev1",
    description="WormTracer package",
    author='Koyo Kuze et al',
    packages=find_packages(),
    license='MIT',
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
