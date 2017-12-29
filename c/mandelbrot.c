/* ******************************************************************** *
 * Copyright (c) 2017 Daniel Küttel                                     *
 *                                                                      *
 * This file is part of fractals.                                       *
 *                                                                      *
 * fractals is free software: you can redistribute it and/or modify     *
 * it under the terms of the GNU General Public License as published by *
 * the Free Software Foundation, either version 3 of the License, or    *
 * (at your option) any later version.                                  *
 *                                                                      *
 * fractals is distributed in the hope that it will be useful,          *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of       *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        *
 * GNU General Public License for more details.                         *
 *                                                                      *
 * You should have received a copy of the GNU General Public License    *
 * along with fractals.  If not, see <http://www.gnu.org/licenses/>.    *
 *                                                                      *
 * ******************************************************************** */

#include <Python.h>
#include "numpy/ndarrayobject.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "numpy/ndarraytypes.h"




static PyMethodDef methods[] = {
	{NULL, NULL, 0, NULL}
};


static void double_uf_juliac(char ** args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
{
	char * out = args[3];
	char * in = args[0];
	char * _maxiter = args[1];
	char * _maxvalue = args[2];

	npy_intp i;
	npy_intp j;
	npy_intp n = dimensions[0];

	npy_intp in_step = steps[0];
	npy_intp out_step = steps[3];

	int maxiter = *((int *) _maxiter);
	double maxvalue = *((double *) _maxvalue);

	Py_complex this_in;

	for(i = 0; i < n; i++)
	{
		this_in = *((Py_complex *) in);

		double z_n_x = 0;
		double z_n_y = 0;

		double swap_x, swap_y;

		for(j = maxiter; j > 0; j--)
		{
			swap_x = z_n_x;
			swap_y = z_n_y;

			z_n_x = swap_x * swap_x - swap_y * swap_y + this_in.real;
			z_n_y = 2 * swap_x * swap_y + this_in.imag;

			if((z_n_x * z_n_x + z_n_y * z_n_y) > maxvalue)
			{
				break;
			}
		}

		*((npy_intp *) out) = maxiter - j;

		in += in_step;
		out += out_step;
	}

}







PyUFuncGenericFunction funcs[1] = {&double_uf_juliac};
static char types[4] = {NPY_CDOUBLE, NPY_INT, NPY_DOUBLE, NPY_INT};
static void *data[1] = {NULL};





static PyModuleDef fractals_backend_mandelbrotmodule = 
{
	PyModuleDef_HEAD_INIT,
	"fractals.backend.mandelbrot",
	"Module containing functions for generating mandelbrot fractals\n",
	-1,
	methods,
	NULL,NULL,NULL,NULL
};

PyMODINIT_FUNC PyInit_mandelbrot(void)
{
	PyObject * module;
	PyObject * juliac;
	PyObject * module_dict;
	module = PyModule_Create(&fractals_backend_mandelbrotmodule);
	if(!module)
	{
		return NULL;
	}

	import_array();
	import_umath();

	juliac = PyUFunc_FromFuncAndData(funcs, data, types, 1, 3, 1, PyUFunc_None,
			"juliac", "calculate closeness to julia sets", 0);

	module_dict = PyModule_GetDict(module);
	PyDict_SetItemString(module_dict, "juliac", juliac);
	Py_DECREF(juliac);

	return module;
}




