# Python template

This repo is meant as a template for new projects.

## Template actions

After using the template action button you have to follow these steps.

### Project Owner
- [ ] Change the repo name
- [ ] Change the meta data in the `pyproject.toml` and `README.md` files
- [ ] Change the version in the `VERSION` file to your starting version, e.g. 0.0.1
- [ ] In the about section, enable the use of the github pages for the documentation
- [ ] After renaming the package, you should run `make sync-venv` and `make requirements` to update the `requirements_dev.txt` file
- [ ] Now you should run `make sync-venv` and then `make pre-commit`
- [ ] Change the module name in the `docs/reference.md` file
- [ ] Once these actions are fulfilled you can delete this template actions section


## Authors
- [Konstantin Göbler](mailto:konstantin.goebler@tum.de)

**Maintainer*:* [Stephan Haug](mailto:stephan.haug@tum.de)
<!-- if this is a student project the maintainer is the supervisor -->

## Table of contents

* [Documentation](#documentation)
* [How to install](#installing)
* [How to build](#building)
* [How to use](#using)
* [How to test](#testing)
* [Github Actions](#actions)

## <a name="documentation">Documentation </a>

Your documentation can be found [here]add link.

## <a name="installing">How to install</a>

The package can be installed locally via:

    pip install -e .

## <a name="building">How to build</a>

Using the [Makefile](Makefile) the package can be installed in an editable way like this:

    make sync-venv

To use the `pre-commit` hooks, one has to enable them in the venv, by

    pre-commit install

Then these hooks are excecuted before every commit. You can run the hooks for all files also separately

    pre-commit run --all-files

or to disable the `pip-compile` hook, which takes some time

    SKIP=pip-compile pre-commit run --all-files

or equivalent

    make pre-commit

## <a name="using">How to use</a>

If possible, a small how-to should be provided here, but the [documentation](#documentation) is probably more appropriate.

## <a name="testing">How to test</a>

The template uses pytest and the test suite can be executed locally via

    python -m pytest

## <a name="actions">Github Actions </a>

### <a name="mkdocs">Documentation with mkdocs </a>

We use mkdocs for building the documentation.

### Pre-commit

With this [workflow](.github/workflows/pre-commit.yml) the pre-commit rules, specified in the [.pre-commit-config.yaml] are executed.

To use pre-commit locally, please use

    pre-commit install

### Testing

With this [workflow](.github/workflows/test_package.yml) the tests are executed.
