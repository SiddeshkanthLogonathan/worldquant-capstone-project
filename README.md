# WorldQuant University Capstone Project: Deep Learning in Multi-period portfolio optimization

## Abstract

Multi-period portfolio optimization is an extension of the single-period MVO prob-
lem. It’s closer to the real deal since multi-period portfolio optimization takes different
time scales, transaction costs, time-varying return forecasts into consideration. Classi-
cal solution to the multi-period portfolio optimization is solving a convex optimization
problem. But with the reduction of computing costs and the rapid development of
artificial intelligence, leverage the deep learning methods in the multi-period portfolio
optimization problem is a possible alternative solution. In our paper, we apply deep
deep learning methods like neural network and deep reinforcement learning approach
to portfolio optimization under multi-period case and comparing the deep learning
approaches with the classical convex optimization approach.


## Prequiates

[Poetry](https://python-poetry.org/docs/) a tool for dependency management and packaging in Python.

### Install Poetry

* Linux, macOS, Windows (WSL)

```
curl -sSL https://install.python-poetry.org | python3 -
```

* Windows (Powershell)

```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

## Run the project

### Install required libraries

```
$ poetry install
$ poetry run poe autoinstall-torch-cuda
```

### Run the project with Poetry

```
$ poetry run jupyter notebook
```

### Export libraries with Poetry

```
$ poetry run pip3 freeze > requirements.txt
```