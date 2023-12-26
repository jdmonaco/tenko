"""
Global run-time state for contexts.
"""

__all__ = ('Tenko',)


import time
import matplotlib as mpl

from roto.dicts import AttrDict
from toolbox.shell import Shell
from toolbox import HOST_DPI


class TenkoState(AttrDict):
    def reset(self):
        for key in self.keys():
            self[key] = None


def get_host_dpi_key(host):
    if host == 'hilbert' and not Shell.clamshell():
        return 'hilbert_retina'
    if host not in HOST_DPI:
        return 'unknown'
    return host


Tenko = TenkoState()


# Handle to most recently created context

Tenko.context = None


# Machine information

Tenko.load_time = time.time()
Tenko.user = Shell.whoami()
Tenko.host = Shell.hostname(short=True)


# Screen resolution

Tenko.screendpi = HOST_DPI[get_host_dpi_key(Tenko.host)]
mpl.rcParams['figure.dpi'] = Tenko.screendpi
