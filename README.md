# composition-based-stability-regression

Use of min, max, std, and mean in e.g., a [RegressorChain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain) to predict `e_above_hull` and measuring the performance improvement over models trained only on the mean.

## Installation
```bash
conda create -n cbsr python==3.9.*
conda activate cbsr
git clone https://github.com/sparks-baird/composition-based-stability-regression.git
cd composition-based-stability-regression
conda install flit
flit install --pth-file # local installation
```
