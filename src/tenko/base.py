"""
Base class for context components.
"""

from numpy import random

from pouty.console import ConsolePrinter
from roto.dicts import hashdict


class TenkoObject(object):

    """
    A base class to provide naming, random seeding, and console functionality.
    """

    __counts = {}

    def __init__(self, name=None, seed=None, color=None, textcolor=None,
        **kwargs):
        """
        Configure object with name, random seed, and console output functions.

        Arguments
        ---------
        name : str
            A unique string name for this object. By default, a name will be
            created from the object's class's name and a count

        seed : str
            A seed-key string to be converted into a random seed for an
            instance-specific RandomState object at the `rnd` attribute

        color : string, valid color name from pouty.conole
            Prefix text color for console output from this object

        textcolor : string, valid color name from pouty.conole
            Message text color for console output from this object

        Any remaining keyword arguments will be consumed but warnings will also
        be issued, since they should typically be consumed earlier in the MRO.
        """
        self._initialized = False

        # Set color/textcolor defaults for TenkoObject instances
        color = 'green' if color is None else color
        textcolor = 'default' if textcolor is None else textcolor

        # Set the class name to an instance attribute
        if hasattr(self.__class__, 'name'):
            self.klass = self.__class__.name
        else:
            self.klass = self.__class__.__name__

        # Set the instance name to a class-based count if unspecified
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            if c == 0:
                self.name = self.klass
            else:
                self.name = f'{self.klass}_{c:03d}'
        else:
            self.name = name

        # Add a ConsolePrinter instance attribute for printing
        if not hasattr(self, 'out'):
            self.out = ConsolePrinter(prefix=self.name, prefix_color=color,
                    message_color=textcolor)

        # Add a debug function
        if not hasattr(self, 'debug'):
            self.debug = self.out.debug

        # Warn about unconsumed kwargs
        for key, value in kwargs.items():
            self.out(f'{key} = {value!r}', prefix='UnconsumedKwargs',
                     warning=True)

        # Set up a per-instance RandomState attribute based on a unique seed
        if not (hasattr(self, 'seed_key') and hasattr(self, 'seed_value') and \
                hasattr(self, 'rnd') and self.seed_key is not None and \
                self.seed_value is not None and self.rnd is not None):
            self.seed_key = None
            self.seed_value = None
            self.rnd = None
            if seed is None and hasattr(type(self), 'spec'):
                specs = dict(self.items())
                specs.update(name=self.name)  # prevent spec collisions
                self.seed_key = hashdict(specs, nchars=32)
            elif seed is not None:
                self.seed_key = seed
            if self.seed_key is not None:
                self.seed_value = sum(list(map(ord, self.seed_key)))
                self.rnd = random.RandomState(seed=self.seed_value)
                self.debug(f'seed = {self.seed_key!r}, value = '
                           f'{self.seed_value}')

        self._initialized = True

    def __repr__(self):
        if hasattr(self, '__qualname__'):
            return self.__qualname__
        if hasattr(self, '__module__') and hasattr(self, '__name__'):
            return f'{self.__module__}{self.__class__.__name__}'
        return object.__repr__(self)

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)
