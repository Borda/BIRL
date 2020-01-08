# Experimentation CIMA dataset

This section is strictly limited to image registration experiment on [CIMA dataset](http://cmp.felk.cvut.cz/~borovji3/?page=dataset).

## Structure

- **Datasets**: the particular dataset setting described in the image/landmarks pairing - csv tables called `dataset_CIMA_<scope>.csv`
- **Script**: the execution script is `run-SOTA-experiments.sh` and it perform all experiments
- **Results**: the experimental results are exported and zipped per particular dataset scope, the archives are `results_size-<scope>.zip`


## References

For complete references see [bibtex](../docs/references.bib).
1. Borovec, J. (2019). BIRL: **Benchmark on Image Registration methods with Landmark validation**. arXiv preprint [arXiv:1912.13452.](https://arxiv.org/abs/1912.13452)