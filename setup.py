from numpy.distutils.core import setup




def configuration(parent_package = '', top_path = None):
	import numpy
	from numpy.distutils.misc_util import Configuration
	from numpy.distutils.misc_util import get_info

	#Necessary for the half-float d-type.
	info = get_info('npymath')

	config = Configuration('',
		parent_package,
		top_path)
	config.add_extension('fractals.backend.mandelbrot',
		['c/mandelbrot.c'], extra_info = info)

	return config


setup(name = "fractals",
	version = "0.0.1",
	description = "A library/program for generating fractals",
	packages = [
		"fractals"
	],
	package_dir = {"fractals": "fractals"},
	configuration = configuration,
	author = "Daniel Knüttel",
	author_email = "daniel.knuettel@daknuett.eu")
