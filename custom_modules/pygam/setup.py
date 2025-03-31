from setuptools import setup, find_packages

setup(
    name='pygam2',
    version='0.1.0',  # Adjust the version as needed
    packages=find_packages(),  # This will automatically find the Python modules in the directory
    install_requires=[  # Add any dependencies your module might need
        'numpy',  # Example, include actual dependencies for pygam
        'scipy',  # Example
    ],
    include_package_data=True,
)
