__version__ = '0.0.1'
__author__ = 'Nico Goernitz'
__date__ = '12.2015'

import numpy as np
import time


global_profiles = dict()  # contains the global profile information


def profile(fn=None):
    """
    This method is a decorator that keeps track of execution times
    and function calls of both, the function itself as well as the
    source code file (which also means that within each file only one
    method with the same name is allowed).
    Does not take care of subdirectories.
    Args:
        fn: decorated function
    Returns: wrapped timer function around 'func'
    """
    skip_first_call = False

    # name of the function
    name = fn.__name__

    # get the name of the file of the function
    fname = fn.__code__.co_filename
    fname = fname[fname.rfind('/')+1:-3]

    # dictionary key.
    # assumes that only one function with name 'name' exists
    # in file 'fname'
    fkey = '{0}'.format(fname)
    key = '{0}'.format(name)

    if global_profiles.has_key(fkey):
        if global_profiles.has_key(fkey):
            fcalls, ftime, fdict = global_profiles[fkey]
            if not fdict.has_key(key):
                fdict[key] = 0, 0., skip_first_call
                global_profiles[fkey] = fcalls, ftime, fdict
    else:
        fdict = dict()
        fdict[key] = 0, 0., skip_first_call
        global_profiles[fkey] = 0, 0., fdict

    def timed(*args, **kw):
        t = time.time()
        result = fn(*args, **kw)
        t = time.time() - t

        fcalls, ftime, fdict = global_profiles[fkey]
        ncalls, ntime, skip = fdict[key]
        if ncalls==0 and skip:
            ntime = 0.
        fdict[key] = ncalls + 1, ntime + t, skip
        global_profiles[fkey] = fcalls + 1, ftime + t, fdict
        return result
    return timed


def print_profiles():
    """
    This function does provide a nice text-based summary of the
    global profile and should therefore be called only once, before the
    programs quits.

    """
    for fkey in  global_profiles.keys():
        fcalls, ftime, fdict = global_profiles[fkey]
        if fcalls==0:
            print('\n-------{0}: unsused.'.format(fkey.ljust(34)))
        else:
            print('\n-------{0}: ncalls={1:3d} total_time={2:1.4f} avg_time={3:1.4f}'.format( \
                fkey.ljust(34), fcalls, ftime, ftime/float(fcalls)))

        keys = fdict.keys()
        times = list()
        for i in range(len(keys)):
            ncalls, ntime, skip = fdict[keys[i]]
            times.append(-ntime)

        sidx = np.argsort(times).tolist()
        for i in sidx:
            ncalls, ntime, skip = fdict[keys[i]]
            if ncalls==0:
                print('      -{0}: unsused.'.format(keys[i].ljust(34)))
            else:
                print('      -{0}: ncalls={1:3d} total_time={2:1.4f} avg_time={3:1.4f}'.format( \
                    keys[i].ljust(34), ncalls, ntime, ntime/float(ncalls)))

