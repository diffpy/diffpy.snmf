Welcome to SNMF's Documentation!
====================================

``SNMF``: Determine structural phase components, their ratios, and how they change with time from
sets of PDFs or powder diffraction patterns.

Introduction
++++++++++++

``snmf`` is a Python package that increases the insight
researchers can obtain from measured atomic pair distribution functions 
(PDFs) or x-ray diffraction patterns (XRD). It was designed to help a
researcher answer the question: "What are the structural signals composing
my chemical reaction and how do they change with time?"

One approach is to take the conventional nonnegative matrix factorization
model and extend it by introducing a stretching factor matrix that accounts
for isotropic stretching of measured signals and returns components that explain
variability beyond this stretching. Conventional nonnegative matrix factorization
(nmf) obtains component signals and weightings of the component signals from a
series of PDF or XRD files. However, this model assumes the components themselves
are constant in time, making this model unable to capture the behavior of

``snmf`` will extend the behavior of nmf by introducing a stretching factor
at each moment in time that isotropically stretches the component signals.
The algorithm will attempt to find three matrices, a "component" matrix that
stores the component signals, a "stretching factor" matrix that stores the
stretching factors of the components at each moment in time, and a "weights"
matrix that stores the weights of each component at each moment in time.
``snmf`` will then plot the components, stretching factors, and weights.

It is important to note that the user must specify the number of component
signals to obtain from the experimental data. Non-physical results may be
obtained if the number of anticipated component signals is too high.

Authors
-------

``snmf`` is developed by members of the Billinge Group at
Columbia University and Brookhaven National Laboratory including 
Ran Gu, Adeolu Ajayi, Qiang Du, and Simon J. L. Billinge.

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
   package

.. include:: ../../CHANGELOG.rst

Indices
-------

* :ref:`genindex`
* :ref:`search`
