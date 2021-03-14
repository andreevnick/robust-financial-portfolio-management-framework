# Copyright 2021 portfolio-guaranteed-framework Authors

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
    """ An abstract interface for Options"""

    @abstractmethod
    def payoff(self, prices, t=None):
        """ Get the value of payoff function for `prices` and time `t`

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

        raise Exception("The method must be defined in a subclass")

    @property
    @abstractmethod
    def expiry(self):
        """ Expiration date"""
        raise Exception("The method must be defined in a subclass")


class EuropeanOption(IOption):
    """ Generic european-style option class

    Parameters
    ----------
    expiry: int
        An expiration date
    payoff_fcn: Callable
        Payoff function for given price(s)
    """

    def __init__(self, expiry: int, payoff_fcn: Callable):
        assert expiry > 0, 'Expiration date must be greater than zero!'
        self._expiry = expiry
        self.payoff_fcn = payoff_fcn

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        if t == self.expiry:
            return self.payoff_fcn(prices)
        return np.full((np.atleast_2d(prices).shape[0],), -np.inf)

    @property
    def expiry(self):
        return self._expiry


class AmericanOption(IOption):
    """ Generic american-style option class

    Parameters
    ----------
    payoff_fcn: Callable
        Payoff function for given price(s)
    """

    def __init__(self, payoff_fcn: Callable):
        self.payoff_fcn = payoff_fcn

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        return self.payoff_fcn(prices)

    @property
    def expiry(self):
        return np.inf


class BermudanOption(IOption):
    """ Bermudan and Canary-styled option class

    Parameters
    ----------
    payoff_dates: array_like
        Payoff dates (as integers)
    payoff_fcn: Callable
        Payoff function for given price(s)


    Notes
    -----
    Bermudan and Canary options are very much alike, so we put both styles in the same class.

    The payoff function is assumed to be independent of time, meaning that for all `t` in `payoff_dates` the payoff function is the same.
    """

    def __init__(self, payoff_dates, payoff_fcn: Callable):
        payoff_dates = np.array(payoff_dates, dtype=int).flatten()
        assert np.all(payoff_dates > 0), 'All payoff dates must be greater than zero!'
        self.payoff_dates = payoff_dates
        self._expiry = self.payoff_dates.max()
        self.payoff_fcn = payoff_fcn

    def payoff(self, prices, t=None):
        prices = np.atleast_2d(prices)
        if t in self.payoff_dates:
            return self.payoff_fcn(prices)
        return np.full((np.atleast_2d(prices).shape[0],), -np.inf)

    @property
    def expiry(self):
        return self._expiry


def make_option(option_type=None, strike=None, payoff_fcn=None, payoff_dates=None):
    """ Create a new option

    You must either give a specific `type` and `strike` or provide an option `payoff_fcn`.

    If `type` and `strike` are given, `payoff_fcn` is ignored.

    If `type` is given, but `strike` is omitted, `strike` defaults to zero.

    If no `type` is given, but `strike` is provided, then `payoff_fcn` must be a wrapper, which accepts `strike` keyword argument and returns a payoff function (Callable) for a given strike.

    If both `type` and `strike` are omitted, `payoff_fcn` must be a function that accepts 1 positional argument.

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

    Returns
    -------
    IOption
        The constructed option

    Examples
    --------

    Create a simple European call option with strike 10 and expiry 4 and evaluate its payoff at diffent points in time.

    >>> from guaranteed.finance import *
    >>> import numpy as np
    >>> def call_10(x):
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

    >>> from guaranteed.finance import *
    >>> import numpy as np
    >>> def call_payoff(strike):
    ...     def call_with_strike(x):
    ...         return np.array(np.maximum((x - strike), np.zeros_like(x)), float).squeeze()
    ...     return call_with_strike
    >>> call = make_option(payoff_fcn = call_payoff, strike=10, payoff_dates=4)
    >>> call.payoff([[9],[11]], 4)
    array([0., 1.])
    >>> call_5 = make_option(payoff_fcn = call_payoff, strike=5, payoff_dates=4)
    >>> call_5.payoff([4,6,10], 4)
    array([0., 1., 5.])

    Alternatively, create the same option as call on max with 1 asset.

    >>> from guaranteed.finance import *
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

    >>> from guaranteed.finance import *
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
        return BermudanOption(payoff_dates, payoff_fcn=payoff_fcn)
    elif isinstance(payoff_dates, int):
        return EuropeanOption(payoff_dates, payoff_fcn=payoff_fcn)
    elif payoff_dates is None:
        return AmericanOption(payoff_fcn=payoff_fcn)
    else:
        raise ValueError('Wrong payoff_dates type ({tp})!'.format(tp=type(payoff_dates)))
