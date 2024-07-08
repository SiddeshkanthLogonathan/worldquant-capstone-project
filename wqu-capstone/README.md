# Run with Poetry

## Introduce Poetry

[Poetry](https://python-poetry.org/docs/) a tool for dependency management and packaging in Python.

### How to install Poetry on your env

* Linux, macOS, Windows (WSL)

```
curl -sSL https://install.python-poetry.org | python3 -
```

* Windows (Powershell)

```
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

### How to add a new library with Poetry

use poetry add commands

```
poetry add pandas_datareader
```


### How to run project with Poetry

```
poetry run jupyter notebook
```