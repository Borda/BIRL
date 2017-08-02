# BIRL: Benchmark on Image Registration methods with Landmark validation

[![Build Status](https://travis-ci.com/Borda/BIRL.svg?token=HksCAm7DV2pJNEbsGJH2&branch=master)](https://travis-ci.com/Borda/BIRL)
[![codecov](https://codecov.io/gh/Borda/BIRL/branch/master/graph/badge.svg?token=JZwA1rlUGA)](https://codecov.io/gh/Borda/BIRL)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9d352210100847629b2c74d2ff3f4993)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Borda/BIRL&amp;utm_campaign=Badge_Grade)
[![Code Health](https://landscape.io/github/Borda/BIRL/master/landscape.svg?style=flat&badge_auth_token=584b288f6a864dcdaf0fcf7f5d392ea8)](https://landscape.io/github/Borda/BIRL/master)
[![Run Status](https://api.shippable.com/projects/585bfa66e18a291000c15f24/badge?branch=master)](https://app.shippable.com/github/Borda/BIRL)
[![Coverage Badge](https://api.shippable.com/projects/585bfa66e18a291000c15f24/coverageBadge?branch=master)](https://app.shippable.com/github/Borda/BIRL)
[![codebeat badge](https://codebeat.co/badges/40d2ac50-5c1b-4b73-86bf-e04080e65010)](https://codebeat.co/a/jirka-borovec/projects/github-com-borda-birl-master)
[![CircleCI](https://circleci.com/gh/Borda/BIRL.svg?style=svg&circle-token=e58b9845aab1b02d749df60060afbac54138ea28)](https://circleci.com/gh/Borda/BIRL)
<!--
[![Codeship Status for Borda/pyImRegBenchmark](https://app.codeship.com/projects/975451b0-b7e1-0134-981a-3617a86d3e20/status?branch=master)](https://app.codeship.com/projects/194566)
-->

This project contains a set of sample images with related landmarks and experimental evaluation of several image registration (state-of-the-art) methods.

As the data we use a [dataset of stain histological tissues](http://cmp.felk.cvut.cz/~borovji3/?page=dataset) image pairs of related (mainly consecutive cuts) sections where each of them in the pair is colour different stain. The apprentice difference and deformations during sensing makes the image registration challenging task.

For evaluation we have set of manually placed landmarks in each image pair at least 40 uniformly spread over the tissue (we do not put any landmarks in backround)

The dataset is defined by CSV file containing pathes to reference and sensed image and their related landmarks _(see `./data_images/list_pairs_imgs_lnds.csv`)_.

![images-landmarks](figures/images-landmarks.jpg)

## Structure

The project contains also a few folders and its brief description is:

* `data_images` - folder with input sample data
    * `images` - contains sample image pairs (reference and sensed one)
    * `landmarks` - contains related landmarks to images in previous folder
* `benchmarks` - directory with benchmark & template and general useful utils
    * `utils` - useful tools and functions
* `bm_experiments` - directory with particular benchmark experiments
* `configs` - configs for registration methods 
* `macros` - macros mainly for ImageJ 
* `scripts` - useful scripts handling some staff around the benchmark itself


## Before benchmarks (pre-processing) 

In the `data_images` folder we provide some sample images with landmarks for registration. This sample registration pairs are saved in `data_images/list_pairs_imgs_lnds.csv`. 

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

* **[bUnwarpJ](http://imagej.net/BUnwarpJ)** is the [ImageJ](https://imagej.nih.gov/ij/) plugin for elastic registration (optional integration with [Feature Extraction](http://imagej.net/Feature_Extraction))
* **[RVSS](http://imagej.net/Register_Virtual_Stack_Slices)** is [ImageJ](https://imagej.nih.gov/ij/) plugin Register Virtual Stack Slices
* **[DROP](http://www.mrf-registration.net/)** is a software for deformable image registration using discrete optimization.
<!-- 
* **[Elastix](http://elastix.isi.uu.nl)** is wide framework for image registration
-->
* ...

### Install methods and run benchmarks

The description how to install and use particular registration methods is described in the documentation (top in each file) for each benchmark experiments.

Experiments on each registration methods can be performed independently with respect to selected parameters and the given set of chosen image pairs.

Sample execution of the "empty" benchmark template
```bash
mkdir results
python benchmarks/bm_template.py \
    -in ./data_images/list_pairs_imgs_lnds.csv \
    -out ./results \
    --unique --visual \
    --an_executable none
```


## License

The project is using the standard [BSD license](http://opensource.org/licenses/BSD-2-Clause).


## References

For complete references see [bibtex](docs/references.bib).
1. J. Borovec, A. Munoz-Barrutia, and J. Kybic, “**Benchmarking of image registration methods for differently stained histological slides**” in IEEE International Conference on Image Processing (ICIP), 2018.


## Appendix - Useful information

### Configure local environment

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

### Running docString tests

documentation and samples of doc string on [pymotw](https://pymotw.com/2/doctest/) and [python/docs](https://docs.python.org/2/library/doctest.html)