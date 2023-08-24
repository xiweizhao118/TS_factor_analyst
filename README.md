# Time Selection: Factor Generation, Combination and Evaluation

<div align=center><img src="https://s2.loli.net/2023/08/11/BgHZjPQVRXsrCw1.png" width = "150" height = "100" />

This is an integrated analysis frame with factor calculation and evaluation. 

<div align=left>

## How to use?

Note that you can add new factor calculation methods and construct your own factor pool, then play with the combination tools and the evaluation tools to renew your factor or to verify its effect. Every tool in this project is open to rebuild, just add your innovative idea and creativity!

### Before a start

This project depends on:

- [akshare](https://github.com/akfamily/akshare)

- [matplotlib](https://github.com/matplotlib/matplotlib)
- [numpy](https://github.com/numpy/numpy)
- [pandas](https://github.com/pandas-dev/pandas)
- [seaborn](https://github.com/mwaskom/seaborn)
- [statsmodels](https://github.com/statsmodels/statsmodels)
- [scipy](https://github.com/scipy/scipy)
- [nri_quant](https://github.com/NovelResearchInvestment/nri_box)
- [torch](https://github.com/torch)

### Start analysis

There are two jupyter notebooks.  `single_factor.ipynb` helps you to analyze a single factor, and `multi_factors.ipynb` integrates tools to load a calculated factor pool and factors selection, combination and evaluation.

Our evaluation method includes:

- distribution plot
- check if distribution is close to normal
- adf test
- ic test
- Grangers causation test
- trading back test
- layered back test

### After analysis

- factors pool loading:
  - Calculated factors are saved as csv files located in the path `./factorloader/data`

- after factor combination:
  - Combination weights are located in the path `./factorcombiner/data`

## Repository Structure

- `./stockdownload` includes downloading stocks' original trading data and its return calculation for the factors combination tool.
- `./factorloader` generates factors pool, all of the factor calculation methods are saved at `factorgens.py` and an instruction also included.
- `./factorcombiner` saves both factors selection and combination tools.
- `./evaluationtools` saves all of the evaluation methods.
- `./utils` includes practical tools and a data's preprocessing tool.

## More instructions

You can check the *handbook.pdf* located at `./Summary presentation`

Further learning materials:

- [利用python对国内各大券商的金工研报进行复现](https://github.com/hugo2046/QuantsPlaybook)
- [择时论文复现笔记](https://www.joinquant.com/user/9df4817f9c39c67ea27e97be2b182d1c)
