"""
BesselML: A Symbolic Regression Framework for Mathematical Functions

This package provides tools for performing symbolic regression on mathematical
functions, with a focus on special functions like hypergeometric functions.
"""

from .main import Problem, Solution, Promising_solution, create_arbitrary_constraint

__version__ = "0.1.0"
__author__ = "Daniel C."
__email__ = "dan.ctvrta@email.cz"

# Explicitly declare what should be imported with "from BesselML import *"
__all__ = ['Problem', 'Solution', 'Promising_solution', 'create_arbitrary_constraint']
