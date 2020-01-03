# Defining loguniform distribution
"""
Credits: Extended from the original definition of rvs function in scipy/scipy/stats/_distn_infrastructure.py
for the class rv_generic and the _rvs function for the uniform distribution from
scipy/scipy/stats/_continuous_distns.py
"""

from scipy.stats import rv_continuous
import numpy as np


class log_uniform_gen(rv_continuous):
    """A log uniform distribution with base 10
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