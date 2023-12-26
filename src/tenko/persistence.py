"""
Automatic caching storage for expensive computed sigals.
"""

import sys

from toolbox.numpy import random
from roto.strings import naturalize
from roto.dicts import hashdict
from tenko.state import Tenko


class AutomaticCache(object):
    """
    Auto-cache computed results with argument-based keys.

    Subclasses override the _compute() method to perform the computation.

    Class attributes
    ----------------

    `_key_params` : tuple of string instance attribute names
        A tuple of argument names to be used as caching keys

    `_data_root` : string
        The root group name under which cached data should be saved

    `_cache_attrs` : tuple of instance attribute names
        A tuple of attribute names that contain the data results to be
        persisted in the datafile cache

    `_save_attrs` : tuple of instance attribute names
        Other attribute names for, e.g., derived scalar quantities that should
        persist through the cache's key-value store

    Note: If `seed=<str>` is provided as a keyword argument, then it is used as
    a seed key to set the state of a numpy.RandomState object which will be
    assigned to the `rnd` instance attribute. If `seed` is not specified, then
    the class name of the object will be used automatically instead.
    """

    _data_root = 'cache'
    _key_params = ('seed',)
    _cache_attrs = ()
    _save_attrs = ()

    def __init__(self, *, seed=None, cache_consume=True, **kwargs):
        """
        Pull key parameter values out of kwargs to be used as cache keys.
        """
        self._initialized = False
        self.__key_values = {}
        if 'seed' not in self._key_params:
            self._key_params = tuple(self._key_params) + ('seed',)

        # Set all key parameter values from kwargs with optional consumption
        for key in self._key_params:
            if key == 'seed':
                continue
            if cache_consume:
                value = kwargs.pop(key, None)
            else:
                value = kwargs.get(key)
            if value is not None:
                self.__key_values[key] = value
            else:
                raise ValueError(f'missing key {key!r}')

        # Set up the random number generator key
        seed = self.__class__.__name__ if seed is None else seed
        self.__key_values['seed'] = sum(map(ord, seed))
        self.__attrs = {k:self.__key_values[k] for k in self._key_params}

        self._finish_init()
        super().__init__(**kwargs)
        self._initialized = True

    def _finish_init(self):
        """
        Finish updating cache paths after construction or loading.
        """
        self.rnd = random.RandomState(seed=self.__key_values['seed'])
        self.__cachename = '{}_{}'.format(naturalize(self.__class__.__name__),
                hashdict(self.__key_values))

    def compute(self):
        """
        Perform the computation and cache the results in context's datafile.
        """
        context = Tenko.context if 'context' in Tenko else None
        if context is not None and self.__check_cache(context):
            return self.__load(context)

        self._compute()

        if context is not None:
            self.__cache(context)

    def _compute(self):
        """
        Subclasses override this method to perform the computation.

        The constructor arguments are available as instance attributes.

        The results of the computation should be set to the instance attributes
        listed in the `_cache_attrs` class attribute. Derived scalar quantities
        should be set to the instance attributes listed in the `_save_attrs`
        class attribute.

        Use the `rnd` instance attribute for random number generation in order
        to use the AutomaticCache random seed handling.
        """
        raise NotImplementedError

    def __cache(self, context):
        """
        Cache the computed data to the context's datafile.
        """
        for name in self._cache_attrs:
            if getattr(self, name) is None:
                raise RuntimeError('run compute() first')

        # Add the scalar attributes to be saved in the cache
        self.__attrs.update({k:getattr(self, k) for k in self._save_attrs})

        # Create the cache data group with all attributes
        grp = context.create_group(self.__cachename, attrs=self.__attrs,
                project=True, root=self._data_root)

        for name in self._cache_attrs:
            data = getattr(self, name)
            context.save_array(data, name, project=True, attrs=self.__attrs,
                               root=grp)

        context.flush_datafile(project=True)

    def __check_cache(self, context):
        """
        Check whether the key specification has been cached.
        """
        return context.has_node(self.__cachename, project=True,
                                root=self._data_root)

    def __load(self, context, cache_path=None):
        """
        Load cached data keyed by the current specification.
        """
        if cache_path is None:
            cpath = context.datapath(self.__cachename, root=self._data_root)
        else:
            cpath = context.datapath(root=cache_path)

        for name in self._cache_attrs:
            setattr(self, name, context.read_array(name, root=cpath,
                    project=True))

        grp = context.get_node(root=cpath, project=True)
        for key in self._key_params:
            self.__key_values[key] = grp._v_attrs[key]
            self.__attrs[name] = grp._v_attrs[key]

        for name in self._save_attrs:
            setattr(self, name, grp._v_attrs[name])
            self.__attrs[name] = grp._v_attrs[name]

        self._finish_init()
        context.out(cpath, prefix='AutoCacheLoad')

    @classmethod
    def load_cache_from_path(cls, cachepath):
        """
        Load a cached object from an existing path ('/<root>/<name>_<hash>').
        """
        context = Tenko.context if 'context' in Tenko else None
        if context is None:
            print('AutoCacheLoadError: no context available', file=sys.stderr)
            return

        obj = cls()
        obj.__load(context, cache_path=cachepath)
        return obj

    def clear_cache(self):
        """
        Remove the cached data for the current specification.
        """
        context = Tenko.context if 'context' in Tenko else None
        if not self.__check_cache(context):
            return

        cachefile = context.get_datafile(readonly=False, project=True)
        cachefile.remove_node('/' + self._data_root, name=self.__cachename)
        cachefile.close()

    def delete_all_cache_data(self):
        """
        Completely delete all cached data that has been stored for this object
        type, regardless of key parameters or specification.

        NOTE: This will wipe out the data root specified as `_data_root`, so
        please make sure there is nothing else also saved in that subtree of
        the data file that you would like to keep!
        """
        context = Tenko.context if 'context' in Tenko else None
        if context is None:
            print('AutoCacheDeleteError: no context available',
                    file=sys.stderr)
            return

        cachefile = context.get_datafile(readonly=False, project=True)
        cachefile.remove_node('/' + self._data_root)
        cachefile.close()
