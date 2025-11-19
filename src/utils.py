#!/usr/bin/env python3

import numpy as np

def normalize(parameters, pmin=None, pmax=None):
    """
    Normalize an array of parameters with respect 
    to a set min and max value or its own min and max values.

    Parameters
    ----------
    parameters : array-like
        An array of parameter values.
    pmin : float, optional
        The minimum at which to normalize with respect to.
        Default is set to min(`parameters`).
    pmax : float, optional
        The maximum at which to normalize with respect to.
        Default is set to max(`parameters`).

    Returns
    -------
    pmin : float
        The minimum of `parameters` if `pmin` is None else `pmin`.
    pmax : float
        The maximum of `parameters` if `pmax` is None else `pmax`
    normed_parameters : array-like
        The normalized set of parameters.
    """
    if pmin is None:
        pmin = np.min(parameters)
    if pmax is None:
        pmax = np.max(parameters)
  
    # don't normalize 1-element data sets
    if pmin == pmax:
        return pmin, pmax, parameters
    
    normed_parameters = 2 * (parameters - pmin) / (pmax - pmin) - 1
    return pmin, pmax, normed_parameters

def denormalize(normed_parameters, pmin, pmax):
    """
    Denormalize a normalized array with respect to a given min and max.

    Parameters
    ----------
    normed_parameters : array-like
        An array of normed parameter values.
    pmin : float
        The minimum at which to denormalize with respect to.
    pmax : float
        The maximum at which to denormalize with respect to.

    Returns
    -------
    parameters : array-like
        The denormalized list of parameters.
    """
    # don't denormalize 1-element data sets
    if pmin == pmax:
        return normed_parameters 

    return (normed_parameters + 1) * (pmax - pmin) / 2 + pmin
