Welcome to SNMF's Documentation!
====================================

``SNMF``: This library implements the stretched non negative matrix factorization (sNMF) and sparse stretched NMF
(ssNMF) algorithms described in the paper "Stretched Non-negative Matrix Factorization" by Ran Gu et al. (2023), 
which is referenced under the Citation section below.

This algorithm is designed to do an NMF factorization on a set of signals ignoring any uniform stretching of the signal
on the independent variable axis. For example, for powder diffraction data taken from samples containing multiple
chemical phases where the measurements were done at different temperatures and the materials were undergoing thermal
expansion.

Introduction
++++++++++++

``snmf`` is a Python package that increases the insight one can obtain from a measured series time-dependent signals
through applying the stretched nonnegative matrix factorization and spare stretched nonnegative matrix factorization
algorithms. The package seeks to answer the question: "What are the structural signals composing my measured signal at
each moment in time?"

One approach is to take the conventional nonnegative matrix factorization model and extend it by introducing a
stretching factor matrix that accounts for isotropic stretching of measured signals and returns components that explain
variability beyond this stretching. Conventional nonnegative matrix factorization (nmf) obtains component signals and
the weightings of the component signals at each moment in time from a series of data. However, this model assumes the
components themselves are constant in time, making this model unable to capture the changing of the components
themselves.

``snmf`` consider components that change with time by introducing a stretching factor for each component at each moment
in time that isotropically stretches the component signals.The algorithm will attempt to find three matrices, a
"component" matrix that stores the component signals, a "stretching factor" matrix that stores the stretching factors of
the components at each moment in time, and a "weights" matrix that stores the weights of each component at each moment
in time. ``snmf`` will then plot the components, stretching factors, and weights.

One import use case of ``snmf`` is for powder diffraction data taken from samples containing multiple
chemical phases where the measurements were done at different temperatures and the materials were undergoing thermal
expansion. The key advantage of ``snmf`` is that it accurately reflects the isotropic change of the atomic distances
within the chemical phases through its addition of stretching factors.

It is important to note that the user must specify the number of component signals to obtain from the experimental data.
Non-physical results may be obtained if the number of anticipated component signals is too high.

Citation
--------

If you use this program for a scientific research that leads
to publication, we ask that you acknowledge use of the program
by citing the following paper in your publication:

   Ran Gu, Yevgeny Rakita, Ling Lan, Zach Thatcher, Gabrielle E. Kamm, Daniel O’Nolan, Brennan Mcbride, Allison Wustrow, James R. Neilson, Karena W. Chapman, Qiang Du, and Simon J. L. Billinge,
   `Stretched Non-negative Matrix Factorization
   <https://doi.org/10.48550/arXiv.2311.15173>`__,
   arXiv:2311.15173 [cond-mat.mtrl-sci] (2023).


Authors
-------

``snmf`` implements the algorithms described in ...., developed by members of the Billinge Group at
Columbia University, Brookhaven National Laboratory, Stony Brook University, Nankai University, and Colorado State
University including Ran Gu, Yevgeny Rakita, Ling Lan, Zach Thatcher, Gabrielle E. Kamm, Daniel O'Nolan, Brennan Mcbride,
Jame R. Neilson, Karena W. Chapman, Qiang Du, and Simon J. L. Billinge.

This software implementation was developed by members of the Billinge Group at Columbia University and Brookhaven
National Laboratory including Ran Gu, Adeolu Ajayi, Qiang Du, and Simon J. L. Billinge.

For a detailed list of contributors, check `here 
<https://github.com/diffpy/diffpy.snmf/graphs/contributors>`_.

To get started, please go to :ref:`quick_start`

.. toctree::
   :maxdepth: 3
   :hidden:

   quickstart

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   license
   release
   Package API <api/diffpy.snmf>

.. include:: ../../CHANGELOG.rst

Indices
-------

* :ref:`genindex`
* :ref:`search`
