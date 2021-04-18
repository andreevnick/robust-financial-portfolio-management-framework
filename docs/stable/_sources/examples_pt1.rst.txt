How-To: Beginners' steps
========================

These examples are also available in :download:`Jupyter Notebook <../examples/basic_usage.ipynb>`

Load package and create basic options
-------------------------------------

>>> import os, sys
>>> sys.path.insert(0, os.path.abspath('/path/to/robustfpm'))
>>> from robustfpm.finance import make_option

Let's start with some basic options:

>>> option1 = make_option(option_type='putonmax', strike=90) # American Put with strike 90
>>> option2 = make_option(option_type='putonmax', strike=80, payoff_dates = 5) # European Put with strike 80 and expiration date 5
>>> option3 = make_option(option_type='callonmax', strike=90, payoff_dates = [3,5]) # Bermudan Call with strike 90 and payoff dates 3 and 5
>>> option4 = make_option(option_type='put2call1') # American Put2Call1 option

Let's see how options got instantiated:

>>> print('Option 1: {}'.format(option1))
>>> print('Option 2: {}'.format(option2))
>>> print('Option 3: {}'.format(option3))
>>> print('Option 4: {}'.format(option4))
Option 1: <robustfpm.finance.derivatives.AmericanOption object at 0x129591a60>
Option 2: <robustfpm.finance.derivatives.EuropeanOption object at 0x1295916d0>
Option 3: <robustfpm.finance.derivatives.BermudanOption object at 0x129591160>
Option 4: <robustfpm.finance.derivatives.AmericanOption object at 0x129591550>

>>> print('Option 1 payoff at x = 80, 90, and 100, t = 3: {}'.format(option1.payoff([[80], [90], [100]], 3)))
>>> print('Option 2 payoff at x = 100, t = 3: {}'.format(option2.payoff(100, 3)))
>>> print('Option 3 payoff at x = 100, t = 3: {}'.format(option3.payoff(100, 3)))
>>> print('Option 3 payoff at x = 100, t = 4: {}'.format(option3.payoff(100, 4)))
Option 1 payoff at x = 80, 90, and 100, t = 3: [10.  0.  0.]
Option 2 payoff at x = 100, t = 3: [-inf]
Option 3 payoff at x = 100, t = 3: [10.]
Option 3 payoff at x = 100, t = 4: [-inf]

Creating solver and solving some problems
-----------------------------------------

Let's create some basic 1D Problem with Rectangular multiplicative dynamics and no trading constraints. For that, we need :class:`robustfpm.pricing.problem.Problem` from :mod:`robustfpm.pricing`:
::

  pm1 = Problem(starting_price=np.array(100),
                    price_dynamics=ConstantDynamics(support=RectangularHandler([.9, 1.1]), type='mult'),
                    trading_constraints=NoConstraints, option=option1,
                    lattice=Lattice(delta=[1]),
                    time_horizon=5)

We begin by instantiating solver with some parameters.

>>> from robustfpm.pricing import *
>>> opts = {'convex_hull_filter': 'qhull', 'convex_hull_prune_fail_count': 0,
>>>         'convex_hull_prune_success_count':0,'convex_hull_prune_corner_n': 3,'convex_hull_prune_seed': 0} 
>>> solver = ConvhullSolver(enable_timer=True, pricer_options=opts, ignore_warnings=True, iter_tick=50)

.. note:: Most of the time, there is no point in tweaking *all* of these parameters, only some, namely :code:`enable_timer` and :code:`iter_tick`.
.. seealso:: :class:`robustfpm.pricing.problem.ConvhullSolver` 


                
Now we solve it and see the result :math:`V_0(x_0)`:

>>> sol1 = solver.solve(pm1)
>>> # the solution is simply a dictionary
>>> print('Value: {0}'.format(sol1['Vf'][0][0]))
Precalculating points for value function evaluation: 0.1079 sec (CPU 0.1079 sec)
Computing value function in the last point: 0.0000 sec (CPU 0.0000 sec)
t = 4
t = 3
iter = 6/67 (8.96%)
iter = 55/67 (82.09%)
t = 2
t = 1
t = 0
Computing value function in intermediate points in time: 0.5200 sec (CPU 0.5200 sec)
Solving the problem: 0.6285 sec (CPU 0.6285 sec)
Value: 5.327786420219619

Let's play around and change Lattice step

>>> pm1.lattice = Lattice(delta=[.1])
>>> sol2 = solver.solve(pm1)
>>> print('Value: {0}'.format(sol2['Vf'][0][0]))
Precalculating points for value function evaluation: 0.6986 sec (CPU 0.6986 sec)
Computing value function in the last point: 0.0002 sec (CPU 0.0002 sec)
t = 4
iter = 1/818 (0.12%)
iter = 170/818 (20.78%)
iter = 283/818 (34.60%)
iter = 314/818 (38.39%)
iter = 320/818 (39.12%)
iter = 347/818 (42.42%)
iter = 392/818 (47.92%)
iter = 461/818 (56.36%)
iter = 497/818 (60.76%)
iter = 513/818 (62.71%)
iter = 525/818 (64.18%)
iter = 567/818 (69.32%)
iter = 572/818 (69.93%)
t = 3
iter = 7/609 (1.15%)
iter = 37/609 (6.08%)
iter = 225/609 (36.95%)
iter = 306/609 (50.25%)
iter = 329/609 (54.02%)
iter = 403/609 (66.17%)
iter = 404/609 (66.34%)
iter = 410/609 (67.32%)
iter = 512/609 (84.07%)
iter = 597/609 (98.03%)
iter = 605/609 (99.34%)
t = 2
iter = 7/405 (1.73%)
iter = 39/405 (9.63%)
iter = 70/405 (17.28%)
iter = 92/405 (22.72%)
iter = 169/405 (41.73%)
iter = 256/405 (63.21%)
iter = 259/405 (63.95%)
iter = 291/405 (71.85%)
iter = 323/405 (79.75%)
iter = 346/405 (85.43%)
t = 1
iter = 51/203 (25.12%)
iter = 107/203 (52.71%)
t = 0
Computing value function in intermediate points in time: 4.2327 sec (CPU 4.2328 sec)
Solving the problem: 4.9321 sec (CPU 4.9322 sec)
Value: 4.415097310413493

Let's try 2D Problem with another option and *additive* dynamics.

>>> pm2 = Problem(starting_price=np.array([91,90]), 
>>>               price_dynamics=ConstantDynamics(support=RectangularHandler([[-1, 1],[-.75, 1]]), type='add'),
>>>               trading_constraints=IdenticalMap(RealSpaceHandler()), option=option4,
>>>               lattice=Lattice(delta=[.1,.1]), time_horizon=5)
>>> solver.iter_tick = 200
>>> sol3 = solver.solve(pm2)
>>> print('Value: {0}'.format(sol3['Vf'][0][0]))
Precalculating points for value function evaluation: 0.9273 sec (CPU 0.9273 sec)
Computing value function in the last point: 0.0005 sec (CPU 0.0004 sec)
t = 4
iter = 357/5913 (6.04%)
iter = 559/5913 (9.45%)
iter = 1361/5913 (23.02%)
iter = 1555/5913 (26.30%)
iter = 1569/5913 (26.53%)
iter = 1777/5913 (30.05%)
iter = 1900/5913 (32.13%)
iter = 2036/5913 (34.43%)
iter = 2090/5913 (35.35%)
iter = 2159/5913 (36.51%)
iter = 2645/5913 (44.73%)
iter = 2840/5913 (48.03%)
iter = 3117/5913 (52.71%)
iter = 3561/5913 (60.22%)
iter = 3707/5913 (62.69%)
iter = 3745/5913 (63.34%)
iter = 3777/5913 (63.88%)
iter = 3839/5913 (64.92%)
iter = 4043/5913 (68.37%)
iter = 4057/5913 (68.61%)
iter = 4127/5913 (69.80%)
iter = 4220/5913 (71.37%)
iter = 4291/5913 (72.57%)
iter = 4525/5913 (76.53%)
iter = 5007/5913 (84.68%)
iter = 5030/5913 (85.07%)
iter = 5103/5913 (86.30%)
iter = 5111/5913 (86.44%)
iter = 5200/5913 (87.94%)
iter = 5314/5913 (89.87%)
iter = 5494/5913 (92.91%)
iter = 5860/5913 (99.10%)
t = 3
iter = 387/3355 (11.54%)
iter = 817/3355 (24.35%)
iter = 829/3355 (24.71%)
iter = 867/3355 (25.84%)
iter = 899/3355 (26.80%)
iter = 1033/3355 (30.79%)
iter = 1207/3355 (35.98%)
iter = 2367/3355 (70.55%)
iter = 2405/3355 (71.68%)
iter = 2468/3355 (73.56%)
iter = 2471/3355 (73.65%)
iter = 2506/3355 (74.69%)
iter = 2844/3355 (84.77%)
iter = 2876/3355 (85.72%)
iter = 2877/3355 (85.75%)
iter = 3120/3355 (93.00%)
iter = 3303/3355 (98.45%)
t = 2
iter = 246/1517 (16.22%)
iter = 676/1517 (44.56%)
iter = 919/1517 (60.58%)
iter = 1072/1517 (70.67%)
iter = 1112/1517 (73.30%)
iter = 1426/1517 (94.00%)
t = 1
t = 0
Computing value function in intermediate points in time: 39.4168 sec (CPU 39.4172 sec)
Solving the problem: 40.3460 sec (CPU 40.3465 sec)
Value: 2.244970331346091


.. note:: :code:`NoConstraints` is just an alias for :code:`IdenticalMap(RealSpaceHandler())`. There is also another alias: :code:`LongOnlyConstraints`, which is just :code:`IdenticalMap(NonNegativeSpaceHandler)`

.. seealso:: 

  :mod:`robustfpm.pricing.multival_map`:
    Module with multivalued mappings, used for both Trading Constraints and Price Dynamics.
  :class:`robustfpm.pricing.multival_map.IMultivalMap`
    Class for multivalued mappings, used for trading constraints
  :class:`robustfpm.pricing.multival_map.PriceDynamics`
    Class for price dynamics
