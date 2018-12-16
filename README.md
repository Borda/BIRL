# BIRL: Benchmark on Image Registration methods with Landmark validation

[![Build Status](https://travis-ci.org/Borda/BIRL.svg?branch=master)](https://travis-ci.org/Borda/BIRL)
[![CircleCI](https://circleci.com/gh/Borda/BIRL.svg?style=svg&circle-token=e58b9845aab1b02d749df60060afbac54138ea28)](https://circleci.com/gh/Borda/BIRL)
[![Build status](https://ci.appveyor.com/api/projects/status/rmfvuxix379eu6fh/branch/master?svg=true)](https://ci.appveyor.com/project/Borda/birl/branch/master)
[![Run Status](https://api.shippable.com/projects/585bfa66e18a291000c15f24/badge?branch=master)](https://app.shippable.com/github/Borda/BIRL)
[![codecov](https://codecov.io/gh/Borda/BIRL/branch/master/graph/badge.svg?token=JZwA1rlUGA)](https://codecov.io/gh/Borda/BIRL)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/b12d7a4a99d549a9baba6c9a83ad6b59)](https://www.codacy.com/project/Borda/BIRL/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/BIRL&amp;utm_campaign=Badge_Grade_Dashboard)
[![Code Health](https://landscape.io/github/Borda/BIRL/master/landscape.svg?style=flat)](https://landscape.io/github/Borda/BIRL/master)
[![codebeat badge](https://codebeat.co/badges/6dd13229-ca9e-4dae-9394-caf5f363082d)](https://codebeat.co/projects/github-com-borda-birl-master)
<!--
[![Coverage Badge](https://api.shippable.com/projects/585bfa66e18a291000c15f24/coverageBadge?branch=master)](https://app.shippable.com/github/Borda/BIRL)
-->

This project contains a set of sample images with related landmarks and experimental evaluation of state-of-the-art image registration methods.

The [dataset of stained histological tissues](http://cmp.felk.cvut.cz/~borovji3/?page=dataset) is composed by image pairs of related sections (mainly, consecutive cuts).
Each image in the pair is coloured with a different stain. 
The registration of those images is a challenging task due to both artifacts and deformations acquired during sample preparation and apparece differences due to staining. 

For evaluation we have manually placed landmarks in each image pair. There are at least 40 uniformly spread over the tissue. 
We do not put any landmarks in the background.

For more information about annotation creation and landmarks handling we refer to the special repository - [Dataset: histology landmarks](http://borda.github.com/dataset-histology-landmarks).

The dataset is defined by a CSV file containing paths to reference and sensed image and their related landmarks _(see `./data_images/pairs-imgs-lnds_mix.csv`)_.

![images-landmarks](figures/images-landmarks.jpg)

## Structure

The project contains the following folders:

* `benchmarks` - package with benchmark & template and general useful utils
    * `utilities` - useful tools and functions
* `bm_dataset` - package handling dataset creation and servicing
* `bm_experiments` - package with particular benchmark experiments
* `data_images` - folder with input sample data
    * `images` - sample image pairs (reference and sensed one)
    * `landmarks` - related landmarks to images in previous folder
    * `lesions_` - samples of histology tissue with annotation
    * `rat-kidney_` - samples of histology tissue with annotation
* `configs` - configs for registration methods 
* `macros_ij` - macros for ImageJ 

## Before benchmarks (pre-processing) 

In the `data_images` folder we provide some sample images with landmarks for registration. 
These sample registration pairs are saved in `data_images/pairs-imgs-lnds_mix.csv`. 

### Prepare Data

There is a script to generate synthetic data. 
Just set an initial image and their corresponding landmarks. 
The script will generate a set of geometrically deformed images mimicking different stains and compute the new related landmarks.

```bash
python scripts/create_real_synth_dataset.py \
    -i ./data_images/images/Rat_Kidney_HE.jpg \
    -l ./data_images/landmarks/Rat_Kidney_HE.csv \
    -o ./output/synth_dataset \
    -nb 5 --nb_jobs 3 --visual
```

When the synthetic datasets have been created, the cover csv file which contains the registration pairs (Reference and Moving image (landmarks)) is generated. 
Two modes are created: _"first2all"_ for registering the first image to all others and _"each2all"_ for registering each image to all other. 
_(note A-B is the same as B-A)_

```bash
python scripts/generate_regist_pairs_all.py \
    -i ./data_images/synth_dataset/*.jpg \
    -l ./data_images/synth_dataset/*.csv \
    -csv ./data_images/cover_synth-dataset.csv \
    --mode each2all
```

## Experiments with included methods

### Included registration methods

* **[bUnwarpJ](http://imagej.net/BUnwarpJ)** is the [ImageJ](https://imagej.nih.gov/ij/) plugin for elastic registration (optional integration with [Feature Extraction](http://imagej.net/Feature_Extraction)).
* ...

### Install methods and run benchmarks

For each benchmark experiment, the explanation about how to install and use a particular registration method is given in the documentation. Brief text at the top of each file.

For each registration method, different experiments can be performed independently using different values of the parameters or image pairs sets. 

Sample execution of the "empty" benchmark template:
```bash
mkdir results
python benchmark/bm_template.py \
    -c ./data_images/pairs-imgs-lnds_mix.csv \
    -o ./results \
    --an_executable none \
    --unique --visual
```

Measure your computer performance using average execution time on several simple image registrations.
The registration consists of loading images, denoising, feature detection, transform estimation and image warping. 
```bash
python bm_experiments/bm_comp_perform.py -o ./results
```
This script generate simple report exported in JSON file on given output path.


### Add custom registration method

[TODO]


## License

The project is using the standard [BSD license](http://opensource.org/licenses/BSD-3-Clause).


## References

For complete references see [bibtex](docs/references.bib).
1. Borovec, J., Munoz-Barrutia, A. & Kybic, J., 2018. **[Benchmarking of image registration methods for differently stained histological slides](https://www.researchgate.net/publication/325019076_Benchmarking_of_image_registration_methods_for_differently_stained_histological_slides)**. In IEEE International Conference on Image Processing (ICIP). Athens. 

## Appendix - Useful information

**Configure local environment**

Create your own local environment, for more information see the [User Guide](https://pip.pypa.io/en/latest/user_guide.html), and install dependencies requirements.txt contains list of packages and can be installed as
```bash
@duda:~$ cd BIRL 
@duda:~/BIRL$ virtualenv env
@duda:~/BIRL$ source env/bin/activate  
(env)@duda:~/BIRL$ pip install -r requirements.txt  
(env)@duda:~/BIRL$ python ...
```
and in the end terminating...
```bash
(env)@duda:~$ deactivate
```

**Running docString tests** - documentation and samples of doc string on [pymotw](https://pymotw.com/2/doctest/) and [python/docs](https://docs.python.org/2/library/doctest.html)

**Listing dataset in command line**  
```bash
find . | sed -e "s/[^-][^\/]*\// |/g" -e "s/|\([^ ]\)/|-\1/" >> dataset.txt
```
