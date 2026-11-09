# FYS-STK4155 Project 2: Learned or chosen features?

Rasul Ruslanovitsj Øzdber, University of Oslo

I build a feed-forward neural network from scratch, with backpropagation and the gradient methods of Project 1. I fit Runge's function f(x) = 1/(1 + 25x²) to noisy data and classify the handwritten digits in MNIST. I compare the network with OLS, Ridge and Lasso for regression, and with logistic regression and scikit-learn for classification, and study how activation functions, depth, width, scaling and regularization affect the results.

## What's in the repository

- `report/`: the report as [PDF](report/report.pdf), its LaTeX source (`report.tex`), references (`References.bib`) and figures
- `Code/`: `network.py`, `optimizers.py`, `regression.py` and the supporting modules, the experiment scripts, three notebooks (`verification.ipynb`, `runge_regression.ipynb` and `mnist_classification.ipynb`), and the tests in `tests/`
- `results/`: the saved numbers (JSON) behind the figures and tables in the report

## How to run

From the root of the repository:

```sh
pip install -r requirements.txt
python -m pytest Code/tests -q
cd Code
python runge_experiments.py
python mnist_experiments.py
python mnist_experiments.py final_evaluation
python verification.py
python figures.py
```

Keep this order, since some experiments read the results of earlier ones. The separate `final_evaluation` command evaluates the selected MNIST models on the test images. MNIST downloads to `data/` the first time it is needed, and the code verifies its checksum. Tested with Python 3.13. The seeds are fixed; results are reproducible with the same machine and library versions, except for timings.

To read the results without rerunning the experiments, open the notebooks in `Code/` and leave `RERUN = False`. They use the saved JSON files to draw the figures. Set `RERUN = True` to repeat the experiments.
