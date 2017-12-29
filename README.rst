fractals
********

.. content::


Supported Fractals
==================

- Mandelbrot fractal `Mandelbrot Set <https://en.wikipedia.org/wiki/Mandelbrot_set>`_

Interface
=========

``fractals`` has two main parts: The backend written in
``C`` for the ``numpy C-API`` and a frontend written in
``python3``.

The backend is used to calculate the numerical background of
the fractal (Or render output, if there is no such
background). It can be accessed using the subpackage
``fractals.backend``.

The frontend has some helper scripts for generating
fractals, including automatic data allocation and
parallelizing the computations.

License
=======

``fractals`` is licensed under the terms of the GNU GPLv3.
