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

This project contains a set of sample images with related landmarks and experimental evaluation of several image registration (state-of-the-art) methods.

As the data we use a [dataset of stain histological tissues](http://cmp.felk.cvut.cz/~borovji3/?page=dataset) image pairs of related (mainly consecutive cuts) sections where each of them in the pair is colour different stain. The apprentice difference and deformations during sensing makes the image registration challenging task.

For evaluation we have set of manually placed landmarks in each image pair at least 40 uniformly spread over the tissue (we do not put any landmarks in backround)

The dataset is defined by CSV file containing paths to reference and sensed image and their related landmarks _(see `./data_images/pairs-imgs-lnds_mix.csv`)_.

![images-landmarks](figures/images-landmarks.jpg)

## Structure

The project contains also a few folders and its brief description is:

* `data_images` - folder with input sample data
    * `images` - contains sample image pairs (reference and sensed one)
    * `landmarks` - contains related landmarks to images in previous folder
* `benchmarks` - directory with benchmark & template and general useful utils
    * `utilities` - useful tools and functions
* `bm_experiments` - directory with particular benchmark experiments
* `configs` - configs for registration methods 
* `macros_ij` - macros mainly for ImageJ 
* `scripts` - useful scripts handling some staff around the benchmark itself


## Before benchmarks (pre-processing) 

In the `data_images` folder we provide some sample images with landmarks for registration. This sample registration pairs are saved in `data_images/pairs-imgs-lnds_mix.csv`. 

### Prepare Data

There is an option to generate synthetic data, such that you set an initial image and landmarks and the script generates  set of geometrical deformed images with also change color space and related computed new landmarks.

```bash
python scripts/create_synth_dataset_real_image.py \
    -img ./data_images/images/Rat_Kidney_HE.jpg \
    -lnd ./data_images/landmarks/Rat_Kidney_HE.csv \
    -out ./output/synth_dataset \
    -nb 5 --nb_jobs 3 --visual
```

When you have generated the synthetic datasets we generate the cover csv file which contains the registration pairs such as Reference and Moving image (landmarks). We generate then in two modes _"first-all"_ for registering the first one to all others and _"all-all"_ for registering each image to all other. 
_(note A-B is the same as B-A so it is the just once)_

```bash
python scripts/create_registration_pairs.py \
    -imgs ./data_images/synth_dataset/*.jpg \
    -lnds ./data_images/synth_dataset/*.csv \
    -csv ./data_images/cover_synth-dataset.csv \
    --mode all-all
```

## Experiments with included methods

### Included registration methods

* **[bUnwarpJ](http://imagej.net/BUnwarpJ)** is the [ImageJ](https://imagej.nih.gov/ij/) plugin for elastic registration (optional integration with [Feature Extraction](http://imagej.net/Feature_Extraction)).
* **[ANTs](https://sourceforge.net/projects/advants)** with variable transformations (elastic, diffeomorphic, diffeomorphisms, unbiased) and similarity metrics (landmarks, cross-correlation, mutual information, etc).
* **[DROP](http://www.mrf-registration.net)** is a software for deformable image registration using discrete optimization.
<!-- 
* **[Elastix](http://elastix.isi.uu.nl)** is wide framework for image registration
* **[RVSS](http://imagej.net/Register_Virtual_Stack_Slices)** is [ImageJ](https://imagej.nih.gov/ij/) plugin Register Virtual Stack Slices
-->
* ...

### Install methods and run benchmarks

The description how to install and use particular registration methods is described in the documentation (top in each file) for each benchmark experiments.

Experiments on each registration methods can be performed independently with respect to selected parameters and the given set of chosen image pairs.

Sample execution of the "empty" benchmark template
```bash
mkdir results
python benchmark/bm_template.py \
    -in ./data_images/pairs-imgs-lnds_mix.csv \
    -out ./results \
    --unique --visual \
    --an_executable none
```


## License

The project is using the standard [BSD license](http://opensource.org/licenses/BSD-2-Clause).


## References

For complete references see [bibtex](docs/references.bib).
1. Borovec, J., Munoz-Barrutia, A. & Kybic, J., 2018. **[Benchmarking of image registration methods for differently stained histological slides](https://www.researchgate.net/publication/325019076_Benchmarking_of_image_registration_methods_for_differently_stained_histological_slides)**. In IEEE International Conference on Image Processing (ICIP). Athens. 

## Appendix - Useful information

**Configure local environment**

Create your own local environment, for more see the [User Guide](https://pip.pypa.io/en/latest/user_guide.html), and install dependencies requirements.txt contains list of packages and can be installed as
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