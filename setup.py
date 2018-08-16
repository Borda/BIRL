"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject

Copyright (C) 2017-2018 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# Always prefer setuptools over distutils
from os import path
from setuptools import setup, find_packages
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'requirements.txt'), encoding='utf-8') as fp:
    requirements = [r.rstrip() for r in fp.readlines() if not r.startswith('#')]

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='birl',
    version='0.2.0',
    description='Benchmark on Image Registration methods'
                ' with Landmark validation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/borda/BIRL',
    author='Jiri Borovec',
    author_email='jiri.borovec@fel.cvut.cz',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='benchmark image registration landmarks',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=requirements,
)
