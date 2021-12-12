# Audio Features

**author:** Eemeli Saari

**email:** eemeli.saari@tuni.fi

---

Main idea is to provide a simple tool to extract features from a given dataset of audio files. Tool is accessable through commandline command `audio_features` after installation.

## Structure

```
img-loader
|
├── setup.py                # Installation script for pip
├── setup.cfg               # Setup configuration file
|
├── audio_pipe              # Python package
│   ├── cli.py              # Commandline tool
│   ├── factors.py          # Component factory
│   ├── pipeline.py         # Pipeline abstractions
│   └── transformers.py     # Feature transformers
|
├── conftest.py             # Pytest fixtures and plugins
├── tests                   # Simple unittests for stability
|
├── summary_report.md       # More detailed description
└── README.md               # This file
```

## Installation

This project uses some of the new Python features so version requirement is +3.8

### 3rd Party Requirements

- [librosa](https://librosa.org/doc/latest/index.html) Audio and music analysis
- [Click](https://click.palletsprojects.com/en/8.0.x/) Commandline tooling
- [Pytest](https://docs.pytest.org/en/6.2.x/) Python testing framework

### Setup

> ! Usage of [virtualenv](https://docs.python.org/3/tutorial/venv.html) or [conda](https://docs.conda.io/en/latest/) is recommended

```bash
(.env)$ pip install .
```

To install test requirements as well

```bash
(.env)$ pip install .[test]
```

## Usage

After installing the tool you can use.

```bash
(.env)$ audio_features run --help
```

### Basics

With only required parameters being path to input folder and output folder

```bash
(.env)$ audio_features run --inpath=genres/ --outpath=features/
```

And to set the configuration to something else

```bash
(.env)$ audio_features --path=other.cfg run --inpath=genres/ --outpath=features/
```

You can also list the available components

```bash
(.env)$ audio_features list-components
```

### Testing

Running tests using pytest on the default data

```
(.env)$ pytest -v
```

## Configuration

The configuration defines `pipelines` that has steps. Each steps output is fed to next steps input. Pipeline consists of different configurable `components`.

### Example

```apacheconf
[pipeline]
steps = [A, B]
type = pipeline

[A]
type = component
name = my_a_factory
param1 = 1
params2 = 5

[B]
type = component
name = my_b_factory
```
