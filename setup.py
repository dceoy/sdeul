#!/usr/bin/env python

from setuptools import find_packages, setup

from sdeul import __version__

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='sdeul',
    version=__version__,
    author='dceoy',
    author_email='dnarsil+github@gmail.com',
    description='Structural Data Extractor using LLMs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dceoy/sdeul.git',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'docopt', 'jsonschema', 'langchain', 'llama-cpp-python'
    ],
    entry_points={
        'console_scripts': ['sdeul=sdeul.cli:main']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development'
    ],
    python_requires='>=3.6',
)
