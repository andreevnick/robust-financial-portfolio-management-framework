# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

"""This submodule consists of different payoff functions for options."""

import numpy as np

__all__ = ['putonmax',
           'callonmax',
           'put2call1']


def putonmax(strike):
    """ Payoff of Put On Max rainbow option generator

    Parameters
    ----------
    strike: float
        Strike price

    Returns
    -------
    Callable
        Payoff function of put on max option for a given strike

    Notes
    -----
    A put on max option gives holder the right to sell the maximum of the risky assets at the strike price at expiry. Its payoff function:

    .. math :: f(x_1, \dots, x_n) = \max(K - \max(x_1, \dots, x_n), 0),

    where K is the strike price of an option

    Examples
    --------

    Get a payoff function of a put on max option on two actives with strike 10. Then, get its value for prices [5,7] and [9,11]

    >>> a = putonmax(10)
    >>> a([[5,7], [9,11]])
    array([3., 0.])
    """

    def putonmax_known_strike(prices, *unused):
        prices = np.atleast_2d(prices)
        return np.maximum(strike - prices.max(axis=1), float(0))

    return putonmax_known_strike


def callonmax(strike):
    """ Payoff of Call On Max rainbow option generator

    Parameters
    ----------
    strike: float
        Strike price

    Returns
    -------
    Callable
        Payoff function of call on max option for a given strike

    Notes
    -----
    A call on max option gives holder the right to purchase the maximum of the risky assets at the strike price at expiry. Its payoff function:

    .. math :: f(x_1, \dots, x_n) = \max(\max(x_1, \dots, x_n) - K, 0),

    where K is the strike price of an option

    Examples
    --------

    Get a payoff function of a call on max option on two actives with strike 10. Then, get its value for prices [5,7] and [9,11]

    >>> a = callonmax(10)
    >>> a([[5,7], [9,11]])
    array([0, 1.])
    """

    def callonmax_known_strike(prices, *unused):
        prices = np.atleast_2d(prices)
        return np.maximum(prices.max(axis=1) - strike, float(0))

    return callonmax_known_strike


def put2call1(prices, *unused):
    """ Payoff of Put 2 Call 1 rainbow option

    Parameters
    ----------
    prices: array_like, size (n,) or (m,n)
        Asset prices

    Returns
    -------
    Callable
        Payoff function of put 2 call 1 option for a given strike

    Notes
    -----
    A put 2 call 1 is an exchange option, giving the right to purchase first asset at the price of the second asset. Its payoff function:

    .. math :: f(x_1, x_2) = \max(x_1 - x_2, 0),

    Examples
    --------

    Get a payoff function of a put 2 call 1 option. Then, get its value for prices [7,5] and [5,7]

    >>> a = put2call1
    >>> a([[7,5], [5,7]])
    array([2., 0.])
    """

    prices = np.atleast_2d(prices)
    return np.maximum(prices[:, 0] - prices[:, 1], float(0))
