from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['reinforcement', 'tensorflow']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(exclude=["tests", "test_integration"]),
    include_package_data=True,
    description='Pythia training application'
)