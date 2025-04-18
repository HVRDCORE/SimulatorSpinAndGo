"""
Optimization utilities for poker simulations.
"""

import functools
from typing import Callable, Any, Optional
import inspect

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define a dummy njit decorator for fallback
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        else:
            return lambda x: x


def njit_if_available(func: Optional[Callable] = None, **kwargs) -> Callable:
    """
    A decorator that applies Numba's njit if available, otherwise returns the original function.
    
    Args:
        func (Optional[Callable]): The function to decorate.
        **kwargs: Additional arguments to pass to njit.
        
    Returns:
        Callable: The decorated function.
    """
    if func is None:
        return lambda f: njit_if_available(f, **kwargs)
    
    if HAS_NUMBA:
        # Check if the function can be compiled with Numba
        sig = inspect.signature(func)
        can_compile = True
        
        # Check for unsupported features
        try:
            source = inspect.getsource(func)
            
            # Classes, closures, and certain operations are not supported
            unsupported = ['class ', 'self.', 'global ', 'nonlocal ']
            for feature in unsupported:
                if feature in source:
                    can_compile = False
                    break
            
            # Only simple types are supported
            for param in sig.parameters.values():
                if param.annotation not in [inspect.Parameter.empty, int, float, bool, str, list, tuple, dict]:
                    can_compile = False
                    break
            
            if can_compile:
                return njit(**kwargs)(func)
        except:
            pass
    
    return func
