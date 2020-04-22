"""A setuptools based setup module.

See:

* https://packaging.python.org/en/latest/distributing.html
* https://github.com/pypa/sampleproject

Copyright (C) 2017-2019 Jiri Borovec <jiri.borovec@fel.cvut.cz>
"""

# Always prefer setuptools over distutils
from os import path
from setuptools import setup
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open

import birl

PATH_HERE = path.abspath(path.dirname(__file__))

with open(path.join(PATH_HERE, 'requirements.txt'), encoding='utf-8') as fp:
    reqs = [rq.rstrip() for rq in fp.readlines()]
    reqs = [ln[:ln.index('#')] if '#' in ln else ln for ln in reqs]
    requirements = [ln for ln in reqs if ln]

# Get the long description from the README file
# with open(path.join(PATH_HERE, 'README.md'), encoding='utf-8') as fp:
#     long_description = fp.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name='BIRL',
    version=birl.__version__,
    url=birl.__homepage__,

    author=birl.__author__,
    author_email=birl.__author_email__,
    license=birl.__license__,
    description='Benchmark on Image Registration methods with Landmark validation',

    long_description=birl.__doc__,
    long_description_content_type='text/markdown',

    packages=['birl', 'birl.utilities'],

    keywords='benchmark image registration landmarks',
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
