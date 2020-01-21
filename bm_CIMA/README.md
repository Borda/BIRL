# Experimentation CIMA dataset

This section is strictly limited to image registration experiment on [CIMA dataset](http://cmp.felk.cvut.cz/~borovji3/?page=dataset).

## Structure

- **Datasets**: the particular dataset setting described in the image/landmarks pairing - csv tables called `dataset_CIMA_<scope>.csv`
- **Script**: the execution script is `run-SOTA-experiments.sh` and it perform all experiments
- **Results**: the experimental results are exported and zipped per particular dataset scope, the archives are `results_size-<scope>.zip`


## Usage

**Reproduce statistic**

You need to unzip the particular result for each dataset scale in to a separate folder (e.g with the same name).
Then  you need to run the [scope notebook](../notebooks/CIMA_SOTA-results_scope.ipynb) for showing results on a particular dataset scope or [comparing notebook](../notebooks/CIMA_SOTA-results_comparing.ipynb) to compare some statistics over two scopes.
Note that with using attached JSON results you do not need to run cells related parsing results from raw benchmarks results.

**Add own method to statistic**

You need to run your benchmark on the particular dataset scope, the image oaring are:
- [10k scope](dataset_CIMA_10k.csv)
- [full scope](dataset_CIMA_full.csv)

Then you can parse just the new results with [evaluation script](../bm_ANHIR/evaluate_submission.py) or execute the parsing cells at the beginning of [scope notebook](../notebooks/CIMA_SOTA-results_scope.ipynb).


## References

For complete references see [bibtex](../docs/references.bib).
1. Borovec, J. (2019). **BIRL: Benchmark on Image Registration methods with Landmark validation**. arXiv preprint [arXiv:1912.13452.](https://arxiv.org/abs/1912.13452)