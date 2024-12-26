## A new kind of perceptron!

Set up env:

```shell
python3 -m venv .venv
source .venv/bin/activate
```

Install pre-commmit:

```shell
brew install pre-commit
```

Setup and install dependencies:

```shell
python setup.py setup
python setup.py install
```

Run `pytest` before committing to make sure you didn't break anything.
