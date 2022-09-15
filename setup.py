from setuptools import setup, find_packages

setup(
    name="just_round",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "gym==0.21",  # Fixed version due to breaking changes in 0.22
        "numpy",
        "torch>=1.11",
        "cloudpickle",
        "pandas",
        "matplotlib",
        "stable_baselines3",
    ],
    description="Supplementary code for 'Just Round: Quantized Observation Spaces Enable Memory Efficient Learning of Dynamic Locomotion'",
    author="Lev Grossman",
    url="https://github.com/A2R-Lab/Just-Round",
    license="MIT",
    python_requires=">=3.7",
)
