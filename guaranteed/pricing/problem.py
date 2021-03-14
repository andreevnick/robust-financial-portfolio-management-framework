# Copyright 2021 portfolio-guaranteed-framework Authors

# Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
# http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
# http://opensource.org/licenses/MIT>, at your option. This file may not be
# copied, modified, or distributed except according to those terms.

""" This submodule implements a :class:`Problem` class that
incapsulates a problem of pricing and hedging an option via a guaranteed approach
and solver classes for solving such problems: :class:`ISolver` as an interface and some realisations."""

import numpy as np
from abc import ABC, abstractmethod
from .multival_map import PriceDynamics, IMultivalMap
from ..finance import IOption
from ..pricing import Grid
from ..util import PTimer, ProfilerData, coalesce

__all__ = ['Problem',
           'ISolver']


class Problem:
    """ A class representing a problem of pricing and hedging an option via a guaranteed approach.

    Parameters
    ----------
    starting_price: np.ndarray, size = (n,)
        Starting prices of assets, as points from :math:`\mathbb{R}^{n}`
    price_dynamics: PriceDynamics
        Price dynamics of a problem
    trading_constraints: IMultivalMap
        A mapping :math:`D_t(\cdot)`, representing trading constraints
    option: IOption
        An option to price (and hedge)
    grid: Grid
        A grid to solve problem on
    time_horizon: int, optional
        Time horizon :math:`N`

    """

    def __init__(self, starting_price, price_dynamics, trading_constraints, option,
                 grid, time_horizon=None, solver=None, profiler_data=None):
        self.starting_price = starting_price
        self.price_dynamics = price_dynamics
        self.trading_constraints = trading_constraints
        self.option = option
        if time_horizon is not int or time_horizon <= 0:
            raise ValueError('Wrong time horizon! Must be a positive integer.')
        self.time_horizon = time_horizon
        self.grid = grid
        self.dimension = grid.delta.shape[0]
        self.solver = solver
        self.profiler_data = coalesce(profiler_data, ProfilerData())
        self.hedge = None
        self.Vp = None
        self.Vx = None

        if time_horizon is None:
            if 'expiry' in self.option.__dict__:
                self.time_horizon = option.expiry
            else:
                raise TypeError('No time horizon and expiration date given! Problem is unbounded.')
        elif 'expiry' in self.option.__dict__ and option.expiry != self.time_horizon:
            # check that if option has an expiration, then it is the same as given.
            raise ValueError(
                'Expiration of an option ({exp_opt}) and given time horizon ({th}) do not match!'.format(
                    exp_opt=option.expiry, th=self.time_horizon
                ))

    def get_precision(self):
        """ Calculates the precision of given problem

        Returns
        -------
        float
            Precision for value function
        """
        # TODO: calculate and return precision via lipshitz constants of price_dynamics and trading_constraints
        raise NotImplementedError('This method is not yet implemented')

    def set_precision(self, precision):
        """ Changes grid to match given precision

        Parameters
        ----------
        precision: float
            target precision
        """
        # TODO: implement
        raise NotImplementedError('This method is not yet implemented')

    def solve(self, calc_hedge=False):
        """ Solves a problem and sets its Vp and Vx (and hedge, if `calc_hedge` is True) to solution

        Parameters
        ----------
        calc_hedge: bool, optional, default = False
            If True, also calculates hedging strategy (might be sufficiently slower)
        """

        if self.solver is None:
            raise ValueError('Solver is not set! Can not solve a problem')
        solution = self.solver.solve(self, calc_hedge)
        if calc_hedge:
            self.Vx, self.Vp, self.hedge = solution
        else:
            self.Vx, self.Vp = solution


class ISolver(ABC):

    @abstractmethod
    def solve(self, problem, calc_hedge=False):
        """ Solves a given problem and calculates hedging strategy (if `calc_hedge` is True)

        Parameters
        ----------
        problem: Problem
            Problem to solve
        calc_hedge: bool, optional, default = False
            If True, also tries to return hedging strategy

        Returns
        -------
        tuple
            First element is values of value function Vx, second is points on `problem.grid` on which Vx is calculated,
             third (optional) is hedging strategy.
        """
        raise NotImplementedError('This method should be implemented!')
