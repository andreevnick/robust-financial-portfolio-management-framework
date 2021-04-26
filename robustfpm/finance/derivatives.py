# Copyright 2021 portfolio-robustfpm-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

"""This submodule implements IOption interface class, European, American, and Bermudan styled option classes.
Also provides option-generating method :func:`make_option` for seamless construction of different options.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from .payoffs import *

__all__ = [
    'IOption',
    'EuropeanOption',
    'AmericanOption',
    'BermudanOption',
    'make_option'
]


class IOption(ABC):
    """
    An abstract interface for Options
    """

    def __init__(self, expiry, payoff_fcn, lipschitz_fcn):
        def const1(t):
            return 1

        if expiry is not None and expiry <= 0:
            raise ValueError('Expiration date must be greater, than zero!')
        if not callable(payoff_fcn):
            raise ValueError('Payoff function must be callable!')
        if lipschitz_fcn is None:
            lipschitz_fcn = const1  # defaults to constant 1
        if not callable(lipschitz_fcn):
            raise ValueError('Lipschitz function must be callable!')
        self._lipschitz_fcn = lipschitz_fcn

    @abstractmethod
    def payoff(self, prices, t=None):
        """
        Get the value of payoff function for `prices` and time `t`

        Parameters
        ----------
        prices: array_like, size = (m,n) or (n,)
            Asset prices (or a set of prices), each price — a point in :math:`\mathbb{R}^{n}`
        t: int
            Current time

        Returns
        -------
        np.ndarray
        """

        raise NotImplementedError("The method must be defined in a subclass")

    @property
    @abstractmethod
    def expiry(self):
        """ Expiration date"""
        raise NotImplementedError("The method must be defined in a subclass")

    @abstractmethod
    def get_lipschitz(self, t):
        r"""
        Return Lipschitz constant for a payoff function at time `t`

        If :math:`g_t(\cdot)` is a payoff function of an option at time `t`, then its Lipschitz constant is such a number :math:`L_{g_t} \in \mathbb{R_+}`, that:

        .. math:: ||g_t(x_1) - g_t(x_0)|| \leqslant L_{g_t}||x_1 - x_0||,\; \forall x_0, x_1

        """
        raise Exception("The method must be defined in a subclass")


class EuropeanOption(IOption):
    """
    Generic european-style option class

    Parameters
    ----------
    expiry: int
        An expiration date
    payoff_fcn: Callable
        Payoff function for given price(s)
    lipschitz_fcn: Callable, default = None
        Function that returns Lipschitz constant for payoff at time `t`, defaults to 1
    """

    def __init__(self, expiry: int, payoff_fcn: Callable, lipschitz_fcn: Callable = None):
        super().__init__(expiry, payoff_fcn, lipschitz_fcn)
        self._expiry = expiry
        self.payoff_fcn = payoff_fcn

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        if t == self.expiry:
            return self.payoff_fcn(prices, t)
        return np.full((np.atleast_2d(prices).shape[0],), -np.inf)

    @property
    def expiry(self):
        return self._expiry

    def get_lipschitz(self, t):
        if t == self.expiry:
            return self._lipschitz_fcn(t)
        return 0


class AmericanOption(IOption):
    """
    Generic american-style option class

    Parameters
    ----------
    payoff_fcn: Callable
        Payoff function for given price(s)
    expiry: int, optional
        Expiration date
    lipschitz_fcn: Callable, default = None
        Function that returns Lipschitz constant for payoff at time `t`, defaults to 1
    """

    def __init__(self, payoff_fcn: Callable, expiry=None, lipschitz_fcn: Callable = None):
        self.payoff_fcn = payoff_fcn
        super().__init__(expiry, payoff_fcn, lipschitz_fcn)
        self._expiry = expiry

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        return self.payoff_fcn(prices, t)

    @property
    def expiry(self):
        return np.inf if self._expiry is None else self._expiry

    def get_lipschitz(self, t):
        return self._lipschitz_fcn(t)


class BermudanOption(IOption):
    """
    Bermudan and Canary-styled option class

    Parameters
    ----------
    payoff_dates: array_like
        Payoff dates (as integers)
    payoff_fcn: Callable
        Payoff function for given price(s)
    lipschitz_fcn: Callable, default = None
        Function that returns Lipschitz constant for payoff at time `t`, defaults to 1


    Notes
    -----
    Bermudan and Canary options are very much alike, so we put both styles in the same class.
    """

    def __init__(self, payoff_dates, payoff_fcn: Callable, lipschitz_fcn: Callable = None):
        payoff_dates = np.array(payoff_dates, dtype=int).flatten()
        super().__init__(payoff_dates.max(), payoff_fcn, lipschitz_fcn)
        if not np.all(payoff_dates > 0):
            raise ValueError('All payoff dates must be greater than zero!')
        self.payoff_dates = payoff_dates
        self._expiry = self.payoff_dates.max()
        self.payoff_fcn = payoff_fcn

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        if t in self.payoff_dates:
            return self.payoff_fcn(prices, t)
        return np.full((np.atleast_2d(prices).shape[0],), -np.inf)

    @property
    def expiry(self):
        return self._expiry

    def get_lipschitz(self, t):
        if t in self.payoff_dates:
            return self._lipschitz_fcn(t)
        return 0


def make_option(option_type=None, strike=None, payoff_fcn=None, payoff_dates=None, lipschitz_fcn=None):
    """
    Create a new option

    You must either give a specific `type` and `strike` or provide an option `payoff_fcn` (and, optionally, `lipschitz_fcn`).

    If `type` and `strike` are given, `payoff_fcn` and `lipschitz_fcn` are ignored.

    If `type` is given, but `strike` is omitted, `strike` defaults to zero.

    If no `type` is given, but `strike` is provided, then `payoff_fcn` must be a wrapper, which accepts `strike` keyword argument and returns a payoff function (Callable) for a given strike.
    The payoff function returned by the wrapper must accept 2 positional arguments, first of which are prices, and the second is time.

    If both `type` and `strike` are omitted, `payoff_fcn` must be a function that accepts 2 positional arguments: prices (first) and time (second).

    Parameters
    ----------
    option_type: str, optional
        A predefined type of option to be constructed.
    strike: float, optional
        Strike price.
    payoff_fcn: Callable, optional
        Payoff function of an option to be constructed.
    payoff_dates: int or list or tuple or np.ndarray, optional
        Expiration date(s). If has only 1 element (or is int), the European option is constructed. If given an array of payoff times — Bermudan. If ommited — American.
    lipschitz_fcn: Callable, optional
        Function that returns Lipschitz constant for `payoff_fcn` at given time. If omitted — defaults to constant 1.

    Returns
    -------
    IOption
        The constructed option

    Examples
    --------

    Create a simple European call option with strike 10 and expiry 4 and __evaluate its payoff at diffent points in time.

    >>> from robustfpm.finance import *
    >>> import numpy as np
    >>> def call_10(x, *usused):
    ...     return np.array(np.maximum((x - 10), np.zeros_like(x)), float).squeeze()
    >>> call = make_option(payoff_fcn = call_10, payoff_dates=4)
    >>> isinstance(call, IOption)
    True
    >>> isinstance(call, EuropeanOption)
    True
    >>> isinstance(call, BermudanOption)
    False
    >>> call.payoff(9, 3)
    array([-inf])
    >>> call.payoff(11, 3)
    array([-inf])
    >>> call.payoff([[9],[11]], 4)
    array([0., 1.])
    >>> call.payoff([9, 11], 4)
    array([0., 1.])

    Do the same with wrapper-function

    >>> from robustfpm.finance import *
    >>> import numpy as np
    >>> def call_payoff(strike):
    ...     def call_with_strike(x, *usused):
    ...         return np.array(np.maximum((x - strike), np.zeros_like(x)), float).squeeze()
    ...     return call_with_strike
    >>> call = make_option(payoff_fcn = call_payoff, strike=10, payoff_dates=4)
    >>> call.payoff([[9],[11]], 4)
    array([0., 1.])
    >>> call_5 = make_option(payoff_fcn = call_payoff, strike=5, payoff_dates=4)
    >>> call_5.payoff([4,6,10], 4)
    array([0., 1., 5.])

    Portfolio of two European call options: the same as with one, but it has a Lipschitz constant equal to 2.

    >>> from robustfpm.finance import *
    >>> import numpy as np
    >>> def call_payoff(strike, *usused):
    ...     def call_with_strike(x):
    ...         return np.array(np.maximum((x - strike), np.zeros_like(x)), float).squeeze()
    ...     return call_with_strike
    >>> def constant_2(t):
    ...     return 2
    >>> call = make_option(payoff_fcn = call_payoff, strike=10, payoff_dates=4, lipschitz_fcn=constant_2)
    >>> call.payoff([[9],[11]], 4)
    array([0., 1.])
    >>> call.get_lipschitz(3)
    0
    >>> call.get_lipschitz(4)
    2

    Alternatively, create the same option as call on max with 1 asset.

    >>> from robustfpm.finance import *
    >>> call = make_option(option_type='callonmax',strike=10, payoff_dates=4)
    >>> isinstance(call, IOption)
    True
    >>> isinstance(call, EuropeanOption)
    True
    >>> call.payoff(11,4)
    array([1.])
    >>> call.payoff([[9],[11]], 4)
    array([0., 1.])
    >>> call.payoff([9, 11], 4)
    array([1.]) #behaves differently since it's actually a call on max

    In the same manner, create a Bermudan and American put2call1 option.

    >>> from robustfpm.finance import *
    >>> Option1 = make_option(option_type='put2call1',payoff_dates=[1,3,5])
    >>> Option2 = make_option(option_type='put2call1')
    >>> isinstance(Option1, BermudanOption)
    True
    >>> isinstance(Option2, AmericanOption)
    True
    >>> Option1.payoff([[1,2],[5,3], [3,5]], 3)
    array([0., 2., 0.])
    >>> Option2.payoff([[1,2],[5,3], [3,5]], 3)
    array([0., 2., 0.])
    >>> Option1.payoff([[1,2],[5,3], [3,5]], 4)
    array([-inf, -inf, -inf])
    >>> Option2.payoff([[1,2],[5,3], [3,5]], 4)
    array([0., 2., 0.])

    """

    if option_type is None:
        # custom payoff function
        if payoff_fcn is None:
            raise TypeError('You must specify either `type` or `payoff_fcn!')
        if strike is not None:
            payoff_fcn = payoff_fcn(strike=strike)
    else:
        # select from known types
        if strike is None:
            # default a strike
            strike = 0
        if option_type == 'putonmax':
            payoff_fcn = putonmax(strike)
        elif option_type == 'callonmax':
            payoff_fcn = callonmax(strike)
        elif option_type == 'put2call1':
            payoff_fcn = put2call1
        else:
            raise ValueError('Unknown option type: {tp}'.format(tp=option_type))
    # now determine option style
    if isinstance(payoff_dates, (list, tuple, np.ndarray)) and len(payoff_dates) == 1:
        payoff_dates = int(payoff_dates[0])
    if isinstance(payoff_dates, (list, tuple, np.ndarray)):
        return BermudanOption(payoff_dates, payoff_fcn=payoff_fcn, lipschitz_fcn=lipschitz_fcn)
    elif isinstance(payoff_dates, int):
        return EuropeanOption(payoff_dates, payoff_fcn=payoff_fcn, lipschitz_fcn=lipschitz_fcn)
    elif payoff_dates is None:
        return AmericanOption(payoff_fcn=payoff_fcn, lipschitz_fcn=lipschitz_fcn)
    else:
        raise ValueError('Wrong payoff_dates type ({tp})!'.format(tp=type(payoff_dates)))
