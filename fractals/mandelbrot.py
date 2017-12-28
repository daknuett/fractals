from .backend.mandelbrot import juliac
from multiprocessing import cpu_count, Pool
import numpy as np
import threading


def _do_get_mandelbrot_window(top_right: complex, bottom_left: complex, 
			impoints: int, repoints: int, maxiter: int, maxvalue: float):
	"""
	Calculate the julia closeness for a given window.
	"""

	x = np.arange(bottom_left.real, top_right.real, (top_right.real - bottom_left.real) / repoints)
	y = 1j * np.arange(bottom_left.imag, top_right.imag,
			(top_right.imag - bottom_left.imag) / impoints,
			dtype = np.complex)

	xx, yy = np.meshgrid(x, y)
	del(x)
	del(y)

	data = xx + yy
	del(xx)
	del(yy)

	return juliac(data, maxiter, maxvalue)

def _do_get_mandelbrot_window_wrapper(args):
	return _do_get_mandelbrot_window(*args)

class Worker(threading.Thread):
	"""
	Currently unused worker class for threading.
	"""
	def __init__(self, func, args):
		threading.Thread.__init__(self)
		self._func = func
		self._args = args
		self.result = None
	def run(self):
		self.result = self._func(*self._args)

def get_mandelbrot_window(top_right: complex, bottom_left: complex,
		impoints: int, repoints: int, maxiter: int, maxvalue: float):
	"""
		Splits the window along the real axis and calculates
		all the subwindows in another process.

		Then all the subwindows are concat'ed to a full sized window.
	"""

	cpus = cpu_count()

	window_size_re = (top_right.real - bottom_left.real) / cpus
	args = [(top_right - (cpus - (i + 1)) * window_size_re, 
				bottom_left + i * window_size_re,
				impoints, repoints / cpus,
				maxiter, maxvalue) for i in range(cpus)]


	with Pool(cpus) as pool:
		return np.concatenate(pool.map(_do_get_mandelbrot_window_wrapper, args), axis = 1)




