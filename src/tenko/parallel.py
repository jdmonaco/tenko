"""
Handle to the parallel client.
"""

__all__ = ('client', 'close')


import ipyparallel as ipp

from pouty import ConsolePrinter

from . import handles


out = ConsolePrinter(prefix="ParallelClient", prefix_color='blue')


def set_default_profile(profile):
    """
    Set the default IPython profile for parallel clients.
    """
    handles.DEFAULT_PROFILE = str(profile)

def client(profile=None):
    """
    Get a parallel client object.
    """
    profile = profile or handles.DEFAULT_PROFILE
    for rc in handles.IPP_CLIENTS:
        if not rc._closed and rc.profile == profile:
            return rc
    rc = ipp.Client(profile=profile)
    handles.IPP_CLIENTS.append(rc)
    out('Attached to \'{}\' cluster with {} engines', rc.profile, len(rc.ids))
    return rc

def close():
    """
    Close the client if it's open.
    """
    profiles = {rc.profile for rc in handles.IPP_CLIENTS if not rc._closed}
    [rc.close() for rc in handles.IPP_CLIENTS]
    handles.IPP_CLIENTS.clear()
    if len(profiles):
        out('Closed parallel client{}: {}', ('','s')[len(profiles)>1],
                str(profiles)[1:-1])
