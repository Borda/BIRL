# Benchmark for image registration methods

This project contains a set of sample images with related landmarks and experimental evaluation of several image registration (state-of-the-art) methods.

As the data we use a dataset of stain histological tissues image pairs of related (mainly consecutive cuts) sections where each of them in the pair is colour different stain. The apprentice difference and deformations during sensing makes the image registration challenging task. 

For evaluation we have set of manually placed landmarks in each image pair at least 40 uniformly spread over the tissue (we do not put any landmarks in backround)

The dataset is defined by CSV file containing pathes to reference and sensed image and their related landmarks (see /data/list_pairs_imgs_lnds.csv).

**Note**: all Python code require Python 2.7 interpreter


## Structure

The project contains also a few folders and its brief description is:

* **data** - folder with input sample data
    * **images** - contains sample image pairs (reference and sensed one)
    * **landmarks** - contains related landmarks to images in previous folder
* **src** - ...
* **mactos** - macros mainly for ImageJ 
* **scripts** - useful scripts handling some staff around the benchmark itself


## Implemented Methods

* **[Elastix](http://elastix.isi.uu.nl)** - general framework for image registration
* **[bUnwarpJ](http://biocomp.cnb.csic.es/~iarganda/bUnwarpJ)** - the ImageJ plugin for elastic registration using SSD metric
* **[RVSS](http://fiji.sc/wiki/index.php/Register_Virtual_Stack_Slices)** - Register Virtual Stack Slices
* ...


## Before benchmark (pre-processing) 

In the **data** folder we provide some sample images with landmarks for registration. This sample registration pairs are saved in **data/list_pairs_imgs_lnds.csv**. 

### Prepare Data

There is an option to generate synthetic data, such that you set an initial image and landmarks and the script generates  set of geometrical deformed images with also change color space and related computed new landmarks.

´´´
python scripts/create_dataset_real_image_synthetic_deformation.py \
    -img data/images/Rat_Kidney_HE.jpg -lnd data/landmarks/Rat_Kidney_HE.csv \
    -out output/synth_dataset -nb 5 --nb_jobs 3 --visu False
´´´

When you have generated the synthetic datasets we generate the cover csv file which contains the registration pairs such as Reference and Moving image (landmarks). We generate then in two modes "1-all" for registering the first one to all others and "all-all" for registering each image to all other (note A-B is the same as B-A so it is the just once).

´´´
python scripts/create_cover_file.py \
    -imgs ../output/synth_dataset/*.jpg -lnds ../output/synth_dataset/*.csv \
    -csv ../output/cover.csv --mode all-all
´´´

## Useful information

### Configure local environment

This benchmark may require specific version of libraries, so we recommend run it in a local environment.

´´´
$ cd benchmark_Registration
~/benchmark_Registration$ virtualenv vEnv
 -> New python executable in pokus/bin/python
 -> Installing setuptools, pip, wheel...done.
~/benchmark_Registration$ source env/bin/activate
(vEnv):~/benchmark_Registration$
´´´

now you are in local environment and you can install packages with specific versions

´´´
pip install SomePackage             # latest version
pip install SomePackage==1.0.4      # specific version
pip install SomePackage>=1.0.4      # minimum version
´´´

file requirements.txt contains list of packages and can be installed as

´(env):~/benchmark_Registration$ pip install -r requirements.txt´
´(env):~/benchmark_Registration$ pip freeze > requirements.txt´

terminating

´(env):~/benchmark_Registration deactivate´

### Running docString tests

documentation and samples of doc string

* https://pymotw.com/2/doctest/
* https://docs.python.org/2/library/doctest.html
