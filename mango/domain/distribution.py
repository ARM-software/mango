# Defining loguniform distribution
"""
Credits: Extended from the original definition of rvs function in scipy/scipy/stats/_distn_infrastructure.py
for the class rv_generic and the _rvs function for the uniform distribution from
scipy/scipy/stats/_continuous_distns.py
"""

from scipy.stats import rv_continuous
# from scipy.stats import *
# import math
# from numpy import (arange, putmask, ravel, ones, shape, ndarray, zeros, floor,
#                    logical_and, log, sqrt, place, argmax, vectorize, asarray,
#                    nan, inf, isinf, NINF, empty)
#
# from scipy._lib._util import check_random_state
import numpy as np


# class log_uniform_gen(rv_continuous):
#
#     def _rvs(self):
#         variable = self._random_state.uniform(0.0, 1.0, self._size)
#         # print('_rvs variable is:',variable)
#         return variable
#
#     def rvs(self, *args, **kwds):
#         """
#         Random variates of given type.
#         Parameters
#         ----------
#         arg1, arg2, arg3,... : array_like
#             The shape parameter(s) for the distribution (see docstring of the
#             instance object for more information).
#         loc : array_like, optional
#             Location parameter (default=0).
#         scale : array_like, optional
#             Scale parameter (default=1).
#         size : int or tuple of ints, optional
#             Defining number of random variates (default is 1).
#         random_state : None or int or ``np.random.RandomState`` instance, optional
#             If int or RandomState, use it for drawing the random variates.
#             If None, rely on ``self.random_state``.
#             Default is None.
#         Returns
#         -------
#         rvs : ndarray or scalar
#             Random variates of given `size`.
#         """
#         discrete = kwds.pop('discrete', None)
#         rndm = kwds.pop('random_state', None)
#         args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
#         cond = logical_and(self._argcheck(*args), (scale >= 0))
#         if not np.all(cond):
#             raise ValueError("Domain error in arguments.")
#
#         if np.all(scale == 0):
#             return loc * ones(size, 'd')
#
#         # extra gymnastics needed for a custom random_state
#         if rndm is not None:
#             random_state_saved = self._random_state
#             self._random_state = check_random_state(rndm)
#
#         # `size` should just be an argument to _rvs(), but for, um,
#         # historical reasons, it is made an attribute that is read
#         # by _rvs().
#         self._size = size
#         vals = self._rvs(*args)
#
#         # print('scale is:',scale,' loc is:',loc)
#
#         # Scale is second parameter, location is first parameter
#         # Logic is uniformly sample the values, then Scale them to the
#         # logarithic values of the scale and loc parameter, and then return
#         # the exponent from that value
#
#         vals = vals * scale + loc
#         shape = vals.shape
#         if shape is np.array(None).shape:
#             # print('vals shape is:',vals.shape)
#             shape = 1
#         else:
#             shape = shape[0]
#
#         if shape > 1:
#             array_10 = np.full((shape), 10)
#             vals = np.power(array_10, vals)
#
#         else:
#             vals = 10 ** vals
#
#         # vals = np.exp(vals)
#         # do not forget to restore the _random_state
#         if rndm is not None:
#             self._random_state = random_state_saved
#
#         # Cast to int if discrete
#         if discrete:
#             if size == ():
#                 vals = int(vals)
#             else:
#                 vals = vals.astype(int)
#
#         return vals
#
#
# # loguniform = log_uniform_gen(a=0.0, b=1.0, name='log_uniform')


class log_uniform_gen(rv_continuous):
    """A log uniform distribution with customizable base
    """

    def __init__(self, *args, **kwargs):
        self.base = 10
        super(log_uniform_gen, self).__init__(*args, **kwargs)

    def _log(self, x):
        return np.log(x) / np.log(self.base)

    def _argcheck(self, a, b):
        return (a > 0) & (b > a)

    def _get_support(self, a, b):
        return a, b

    def _pdf(self, x, a, b):
        # reciprocal.pdf(x, a, b) = 1 / (x*log(b/a))
        return 1.0 / (x * self._log(b * 1.0 / a))

    def _logpdf(self, x, a, b):
        return np.log(x) - np.log(self._log(b * 1.0 / a))

    def _cdf(self, x, a, b):
        return (self._log(x) - self._log(a)) / self._log(b * 1.0 / a)

    def _ppf(self, q, a, b):
        return a*pow(b*1.0/a, q)

    def _munp(self, n, a, b):
        return 1.0/self._log(b*1.0/a) / n * (pow(b*1.0, n) - pow(a*1.0, n))

    def _entropy(self, a, b):
        return 0.5*np.log(a*b)+np.log(self._log(b*1.0/a))


loguniform = log_uniform_gen(name='loguniform')