# Benchmark for image registration methods

This project contains a set of sample images with related landmarks and experimental evaluation of several image registration (state-of-the-art) methods.

As the data we use a dataset of stain histological tissues image pairs of related (mainly consecutive cuts) sections where each of them in the pair is colour different stain. The apprentice difference and deformations during sensing makes the image registration challenging task. 

For evaluation we have set of manually placed landmarks in each image pair at least 40 uniformly spread over the tissue (we do not put any landmarks in backround)

The dataset is defined by CSV file containing pathes to reference and sensed image and their related landmarks (see /data/list_pairs_imgs_lnds.csv).

**Note**: all Python code require Python 2.7 interpreter

## Configure local environment

This benchmark may require specific version of libraries, so we recommend run it in a local environment.

´´´
$ cd benchmark_Registration
~/benchmark_Registration$ virtualenv env
 -> New python executable in pokus/bin/python
 -> Installing setuptools, pip, wheel...done.
~/benchmark_Registration$ source env/bin/activate
(env):~/benchmark_Registration$
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


## Structure

The project contains also a few folders and its brief description is:

* **data** - folder with input sample data
    * **images** - contains sample image pairs (reference and sensed one)
    * **landmarks** - contains related landmarks to images in previous folder
* **src** - ...
* ...


## Implemented Methods

* **[Elastix](http://elastix.isi.uu.nl)** - general framework for image registration
* **[bUnwarpJ](http://biocomp.cnb.csic.es/~iarganda/bUnwarpJ)** - the ImageJ plugin for elastic registration using SSD metric
* **[RVSS](http://fiji.sc/wiki/index.php/Register_Virtual_Stack_Slices)** - Register Virtual Stack Slices
* ...


