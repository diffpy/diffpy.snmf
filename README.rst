|Icon| |title|_
===============

.. |title| replace:: diffpy.snmf
.. _title: https://diffpy.github.io/diffpy.snmf

.. |Icon| image:: https://avatars.githubusercontent.com/diffpy
        :target: https://diffpy.github.io/diffpy.snmf
        :height: 100px

|PyPi| |Forge| |PythonVersion| |PR|

|CI| |Codecov| |Black| |Tracking|

.. |Black| image:: https://img.shields.io/badge/code_style-black-black
        :target: https://github.com/psf/black

.. |CI| image:: https://github.com/diffpy/diffpy.snmf/actions/workflows/matrix-and-codecov-on-merge-to-main.yml/badge.svg
        :target: https://github.com/diffpy/diffpy.snmf/actions/workflows/matrix-and-codecov-on-merge-to-main.yml

.. |Codecov| image:: https://codecov.io/gh/diffpy/diffpy.snmf/branch/main/graph/badge.svg
        :target: https://codecov.io/gh/diffpy/diffpy.snmf

.. |Forge| image:: https://img.shields.io/conda/vn/conda-forge/diffpy.snmf
        :target: https://anaconda.org/conda-forge/diffpy.snmf

.. |PR| image:: https://img.shields.io/badge/PR-Welcome-29ab47ff

.. |PyPi| image:: https://img.shields.io/pypi/v/diffpy.snmf
        :target: https://pypi.org/project/diffpy.snmf/

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/diffpy.snmf
        :target: https://pypi.org/project/diffpy.snmf/

.. |Tracking| image:: https://img.shields.io/badge/issue_tracking-github-blue
        :target: https://github.com/diffpy/diffpy.snmf/issues

A python package implementing the stretched NMF algorithm.

``diffpy.snmf`` implements the stretched non negative matrix factorization (sNMF) and sparse stretched NMF
(ssNMF) algorithms.

This algorithm is designed to do an NMF factorization on a set of signals ignoring any uniform stretching of the signal
on the independent variable axis. For example, for powder diffraction data taken from samples containing multiple
chemical phases where the measurements were done at different temperatures and the materials were undergoing thermal
expansion.

For more information about the diffpy.snmf library, please consult our `online documentation <https://diffpy.github.io/diffpy.snmf>`_.

Citation
--------

If you use this program for a scientific research that leads
to publication, we ask that you acknowledge use of the program
by citing the following paper in your publication:

   Ran Gu, Yevgeny Rakita, Ling Lan, Zach Thatcher, Gabrielle E. Kamm, Daniel O’Nolan, Brennan Mcbride, Allison Wustrow, James R. Neilson, Karena W. Chapman, Qiang Du, and Simon J. L. Billinge,
   `Stretched Non-negative Matrix Factorization
   <https://doi.org/10.1038/s41524-024-01377-5>`__,
   *npj Comput Mater* **10**, 193 (2024).


Installation
------------

The preferred method is to use `Miniconda Python
<https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_
and install from the "conda-forge" channel of Conda packages.

To add "conda-forge" to the conda channels, run the following in a terminal. ::

        conda config --add channels conda-forge

We want to install our packages in a suitable conda environment.
The following creates and activates a new environment named ``diffpy.snmf_env`` ::

        conda create -n diffpy.snmf_env python=3
        conda activate diffpy.snmf_env

Then, to fully install ``diffpy.snmf`` in our active environment, run ::

        conda install diffpy.snmf

Another option is to use ``pip`` to download and install the latest release from
`Python Package Index <https://pypi.python.org>`_.
To install using ``pip`` into your ``diffpy.snmf_env`` environment, type ::

        pip install diffpy.snmf

If you prefer to install from sources, after installing the dependencies, obtain the source archive from
`GitHub <https://github.com/diffpy/diffpy.snmf/>`_. Once installed, ``cd`` into your ``diffpy.snmf`` directory
and run the following ::

        pip install .

Support and Contribute
----------------------

`Diffpy user group <https://groups.google.com/g/diffpy-users>`_ is the discussion forum for general questions and discussions about the use of diffpy.snmf. Please join the diffpy.snmf users community by joining the Google group. The diffpy.snmf project welcomes your expertise and enthusiasm!

If you see a bug or want to request a feature, please `report it as an issue <https://github.com/diffpy/diffpy.snmf/issues>`_ and/or `submit a fix as a PR <https://github.com/diffpy/diffpy.snmf/pulls>`_. You can also post it to the `Diffpy user group <https://groups.google.com/g/diffpy-users>`_.

Feel free to fork the project and contribute. To install diffpy.snmf
in a development mode, with its sources being directly used by Python
rather than copied to a package directory, use the following in the root
directory ::

        pip install -e .

To ensure code quality and to prevent accidental commits into the default branch, please set up the use of our pre-commit
hooks.

1. Install pre-commit in your working environment by running ``conda install pre-commit``.

2. Initialize pre-commit (one time only) ``pre-commit install``.

Thereafter your code will be linted by black and isort and checked against flake8 before you can commit.
If it fails by black or isort, just rerun and it should pass (black and isort will modify the files so should
pass after they are modified). If the flake8 test fails please see the error messages and fix them manually before
trying to commit again.

Improvements and fixes are always appreciated.

Before contribuing, please read our `Code of Conduct <https://github.com/diffpy/diffpy.snmf/blob/main/CODE_OF_CONDUCT.rst>`_.

Contact
-------

For more information on diffpy.snmf please visit the project `web-page <https://diffpy.github.io/>`_ or email Prof. Simon Billinge at sb2896@columbia.edu.
