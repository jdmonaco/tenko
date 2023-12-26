"""
Managed context for data analysis runs.
"""

import os
try:
    import ipdb as pdb
except ImportError:
    import pdb
import sys
import time
try:
    import simplejson as json
except ImportError:
    import json
import inspect
import datetime
import subprocess
import tempfile
from importlib import import_module
from decorator import decorator
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
import pandas as pd

from toolbox import HOME, PROJDIR, IMACPRO_DPI
from toolbox.shell import Shell
from pouty.anybar import AnyBar
from pouty.console import ConsolePrinter, COL_FUNC
from roto import datapath as tpath
from roto.data import DataStore
from roto.figures import get_svg_figinfo
from roto.paths import uniquify, tilde
from roto.strings import snake2title, sluggify, naturalize
from roto.dicts import AttrDict, merge_two_dicts, hashdict
from specify import is_specified, is_param

from . import parallel
from .state import Tenko
from .base import TenkoObject


CALLFILE = 'call.log'
INITFILE = 'init.json'
ENVFILE = 'env.json'


class Context(AttrDict):
    pass


@decorator
def step(_f_, *args, **kwargs):
    """
    Declare a method as a compute step in this context.
    """
    self = args[0]
    status = { 'OK': False }
    self._step_enter(_f_, args, kwargs)
    res = self._step_execute(_f_, args, kwargs, status)
    self._step_exit(_f_, args, kwargs, status)
    return res


class AbstractBaseContext(TenkoObject):

    """
    Smart, self-documenting, all-inclusive context for reproducible analysis.
    """

    def _arg(self, name, value, dflt=None, norm=False, path=False,
        optional=False):
        """
        Return non-null keyword values or the class attribute-based default.
        """
        if value is not None:
            if path:
                return os.path.abspath(value)
            if norm:
                return sluggify(value)
            return value
        if dflt is not None:
            if path:
                return os.path.abspath(dflt)
            if norm:
                return sluggify(dflt)
            return dflt
        if name is not None:
            clsname = f'_{name}'
            if not hasattr(self.__class__, clsname):
                return None
            cls_dflt = getattr(self.__class__, clsname)
            if cls_dflt is not None:
                if path:
                    return os.path.abspath(cls_dflt)
                if norm:
                    return sluggify(cls_dflt)
                return cls_dflt
        if not optional:
            print(f'Warning: missing value for \'{name}\'', file=sys.stderr)
        return None

    def __init__(self   , desc=None    , tag=None       , projname=None ,
        version=None    , repodir=None , rootdir=None   , datadir=None  ,
        resdir=None     , regdir=None  , moduledir=None , h5file=None   ,
        h5proj=None     , ctxdir=None  , admindir=None  , tmpdir=None   ,
        rundir=None     , profile=None , logcolor=None  , figfmt=None   ,
        staticfigs=None , dryrun=None  , **kwargs):
        # Set the text color and initialize console output
        self._logcolor = self._arg('logcolor', logcolor, norm=True)
        super().__init__(color=self._logcolor, **kwargs)

        # Set key identifying attributes for this context
        self._name = self._arg('__name__', self.__class__.__name__, norm=True)
        self._desc = self._arg('desc', desc, norm=True, optional=True)
        self._tag = self._arg('tag', tag, norm=True, optional=True)
        self._projname = self._arg('projname', projname, norm=True)
        self._version = self._arg('version', version, norm=True)
        self._repodir = self._arg('repodir', repodir, path=True)

        # For a dry run, use a temporary directory as the project root
        _rootdflt = os.path.join(PROJDIR, self._projname or 'toolbox')
        self._dryrun_tempdir = None
        self._dryrun = self._arg('dryrun', dryrun, dflt=False, optional=True)
        if self._dryrun:
            self._dryrun_tempdir = tempfile.TemporaryDirectory(prefix='tenko_')
            _rootdflt = self._dryrun_tempdir.name
        self._rootdir = self._arg('rootdir', rootdir, dflt=_rootdflt, path=True)

        # Set up the folder tree for analysis files and output
        self._datadir = self._arg('datadir', datadir, dflt=os.path.join(
            self._rootdir, 'data'))
        self._resdir = self._arg('resdir', resdir, dflt=os.path.join(
            self._rootdir, 'results'))
        self._regdir = self._arg('regdir', regdir, path=True, optional=True)
        self._modname = self.__class__.__module__.split('.')[-1]
        self._moduledir = self._arg('moduledir', moduledir, dflt=os.path.join(
            self._rootdir, self._modname))
        ctxdflt = os.path.join(self._moduledir, self._version)
        if self._desc is not None: ctxdflt += f'-{self._desc}'
        if self._tag is not None: ctxdflt += f'+{self._tag}'
        self._ctxdir = self._arg('ctxdir', ctxdir, dflt=ctxdflt)
        self._admindir = self._arg('admindir', admindir, dflt=os.path.join(
            self._ctxdir, 'admin'))
        self._tmpdir = self._arg('tmpdir', tmpdir, dflt=os.path.join(
            self._ctxdir, 'temp'))
        self._rundir = self._arg('rundir', rundir, dflt=self._tmpdir)

        # Check for existing context data and load it if available
        loaded = False
        if os.path.isdir(self._admindir):
            if os.path.isfile(os.path.join(self._admindir, INITFILE)):
                self.load(self, load_instance=True)
                loaded = True
        if loaded: self.hline()

        # Process other arguments not needed for the admin load after the load
        self._profile = self._arg('profile', profile, dflt=self._projname)
        self._figfmt = self._arg('figfmt', figfmt, dflt='mpl')
        self._staticfigs = self._arg('staticfigs', staticfigs, dflt=True)
        self._h5file = self._arg('h5file', h5file, dflt=os.path.join(
            self._moduledir, f'{self._name}.h5'))
        self._h5proj = self._arg('h5proj', h5proj, dflt=os.path.join(
            self._datadir, f'{self._projname}.h5'))

        # Create the HDF datafile handlers for the context and project
        self._datafile = None
        self._projfile = None
        self._set_datafile(self._h5file)
        self._set_project_datafile(self._h5proj)

        # Set the ipyparallel profile
        self.set_parallel_profile(self._profile)

        # Load the persistent namespace context with key/attribute access
        self.c = Context()
        self._load_context()

        # Context state variables
        self._figures = {}
        self._savefig = {}
        self._lastfig = None
        self._savefig_path = None
        self._holdfigs = False
        self._lastcall = None
        self._running = False
        self._anybar = None

        # Finished initializing!
        Tenko.context = self
        self._save()

    def pprint(self):
        """
        Print out a colorful listing of context info including all paths.
        """
        col_w = 13
        s = ['Class:'.ljust(col_w) + self.__class__.__name__]
        s += ['Module:'.ljust(col_w) + self.__class__.__module__]
        if self._desc:
            s += ['Description:'.ljust(col_w) + f"'{self._desc}'"]
        if self._tag:
            s += ['Tag:'.ljust(col_w) + f"'{self._tag}'"]
        s += ['ProjectDir:'.ljust(col_w) + tilde(self._rootdir)]
        s += ['DataDir:'.ljust(col_w) + tilde(self._datadir)]
        if self._projfile.exists():
            s += ['ProjectData:'.ljust(col_w) + tilde(self._projfile.path())]
        s += ['ResultsDir:'.ljust(col_w) + tilde(self._resdir)]
        if self._regdir:
            s += ['RegDir:'.ljust(col_w) + tilde(self._regdir)]
        s += ['ModuleDir:'.ljust(col_w) + tilde(self._moduledir)]
        s += ['ContextDir:'.ljust(col_w) + tilde(self._ctxdir)]
        if self._datafile.exists():
            s += ['ContextData:'.ljust(col_w) + tilde(self._datafile.path())]
        env_keys = self.c.keys()
        if env_keys:
            s += ['EnvKeys:'.ljust(col_w) + ', '.join(env_keys)]
        self.printf('\n'.join(s) + '\n')

    # Namespace methods

    def get_global_scope(self):
        """
        Helper method to access the global scope of the context.
        """
        return import_module(self.__class__.__module__).__dict__

    def get_config(self):
        """
        Helper method to access the shared configuration for the context.
        """
        cfg = '.'.join(self.__class__.__module__.split('.')[:-1] + ['config'])
        return import_module(cfg).Config

    # Key-value persistence

    def get_json(self, *path, return_path=True):
        """
        Search for a JSON file in context directories and return (path, data).

        Note: The '.json' extension is automatically added if omitted.
        """
        fpath = os.path.join(*path)
        if not fpath.endswith('.json'):
            fpath += '.json'

        if os.path.isabs(fpath):
            if os.path.isfile(fpath):
                path = fpath
            else:
                self.out(fpath, prefix='MissingJSONFile', error=True)
                return
        else:
            path = self.search(fpath)
            if path is None:
                return

        data = self.read_json(path)

        if not return_path:
            return data
        return path, data

    def read_json(self, *path, base=None):
        """
        Read key-value data from JSON file at the specified path.

        Note: The '.json' extension is automatically added if omitted.
        """
        fpath = self.path(*path, base=base)
        if not fpath.endswith('.json'):
            fpath += '.json'

        with open(fpath, 'r') as fd:
            data = json.load(fd)
        return data

    def write_json(self, data, *path, base=None, sort=False, unique=False):
        """
        Save key-value data to JSON file at the specified path.

        Note: The '.json' extension is automatically added if omitted.
        """
        p = os.path.join(*path)
        if not p:
            self.out('No path specified', prefix='WriteJSON', error=True)
            raise ValueError('empty path')
        if not p.endswith('.json'):
            p = f'{p}.json'
        fpath = self.path(p, base=base, unique=unique)

        with open(fpath, 'w') as fd:
            json.dump({k:v for k,v in data.items() if v is not None},
                       fd, indent=2, skipkeys=True, sort_keys=sort,
                       separators=(', ', ': '))
        return fpath

    def _save_context(self):
        """
        Save the persistent key-value store.
        """
        self.write_json(self.c, ENVFILE, base='admin')

    def _load_context(self):
        """
        Load the persistent key-value store if it exists.
        """
        sfn = self.path(ENVFILE, base='admin')
        if not os.path.isfile(sfn): return
        self.c.update(self.read_json(sfn))

    # Load/save methods

    @classmethod
    def load(cls, instance_or_context_dir, load_instance=False):
        """Return a new context for a previous run directory."""
        out = ConsolePrinter(prefix="{}Loader".format(cls.__name__),
                prefix_color='brown')

        if hasattr(instance_or_context_dir, '_admindir'):
            inst = instance_or_context_dir
            admindir = instance_or_context_dir._admindir
        elif type(instance_or_context_dir) is str:
            inst = None
            admindir = os.path.join(instance_or_context_dir, 'admin')
        else:
            out('Requires a context object or admin path',
                    prefix='BadArgument', error=True)
            return

        initpath = os.path.join(admindir, INITFILE)
        if not os.path.isfile(initpath):
            out(initpath, prefix='MissingFile', error=True)
            return

        try:
            with open(initpath, 'r') as fd:
                initargs = json.load(fd)
        except ValueError:
            out(initpath, prefix='InvalidJSON', error=True)
            return
        else:
            out(os.path.split(admindir)[0], prefix='LoadedContext')

        if load_instance and inst is not None:
            for k, v in initargs.items():
                setattr(inst, f'_{k}', v)
            return inst
        return cls(**initargs)

    def _save(self):
        """
        Save the constructor parameters for this object.
        """
        self.write_json({
            'desc'       : self._desc,
            'tag'        : self._tag,
            'projname'   : self._projname,
            'version'    : self._version,
            'repodir'    : self._repodir,
            'rootdir'    : self._rootdir,
            'datadir'    : self._datadir,
            'resdir'     : self._resdir,
            'regdir'     : self._regdir,
            'moduledir'  : self._moduledir,
            'h5file'     : self._datafile.path(),
            'h5proj'     : self._projfile.path(),
            'ctxdir'     : self._ctxdir,
            'admindir'   : self._admindir,
            'tmpdir'     : self._tmpdir,
            'rundir'     : self._rundir,
            'profile'    : self._profile,
            'logcolor'   : self._logcolor,
            'figfmt'     : self._figfmt,
            'staticfigs' : self._staticfigs,
        }, INITFILE, base='admin')

    def register(self):
        """
        Link this context into the results directory.
        """
        self.close_logfile()
        self.close_datafile()

        # Register as: <version>[-<desc>]+<class-name>[+<tag>]
        resdest = os.path.join(self._resdir, self._version)
        if self._desc:
            resdest += f'-{self._desc}'
        resdest += f'+{self._name}'
        if self._tag:
            resdest += f'+{self._tag}'

        if os.path.isdir(resdest):
            self.out('Link already exists: {}', resdest,
                    error=True)
            return

        if not os.path.isdir(self._resdir):
            os.makedirs(self._resdir)

        try:
            os.symlink(self._ctxdir, resdest, target_is_directory=True)
        except IOError:
            self.out(resdest, prefix='FailedLink', error=True)
            return

        # Hard link the data file into the context directory
        if self._datafile:
            parent, name = os.path.split(self._datafile.path())
            if os.path.abspath(parent) == self._moduledir:
                new_h5file = os.path.join(self._ctxdir, name)
                os.link(self._h5file, new_h5file)
                self.set_datafile(new_h5file)

        # Set/save the registration path
        self._regdir = resdest
        self._save()

        self.out(self._regdir, prefix='Registration')

    # Random number seeds

    def set_default_random_seed(self, seed=None, key=None):
        """
        Set the default numpy random seed from a string seed or hash.
        """
        if seed is None:
            key = self.seed_key
            seed = self.seed_value  # set by TenkoObject init for all Specified
        else:
            key = hashdict(dict(seed=seed), nchars=32)
            seed = sum(list(map(ord, key)))

        # Set the default numpy random state with the seed
        np.random.seed(seed)
        self.out(f'Default RNG seed = {seed}')

    # Run directory path methods

    def path(self, *path, base=None, unique=False):
        """
        Get an absolute path in the context directory structure.

        Keyword `base` may be 'admin', 'run', 'temp', 'context', 'module',
        'root', 'data', or 'results' (default 'run').
        """
        base = 'run' if base is None else base
        if path and os.path.isabs(path[0]):
            fp = os.path.join(*path)
            if unique:
                fp = uniquify(*os.path.splitext(fp))
            return fp

        try:
            root = dict(
                        admin = self._admindir,
                        run = self._rundir,
                        temp = self._tmpdir,
                        context = self._ctxdir,
                        module = self._moduledir,
                        root = self._rootdir,
                        data = self._datadir,
                        results = self._resdir,
                    )[base]
        except KeyError:
            self.out(base, prefix='UnknownDirType', error=True)
            return os.path.join(*path)

        if not os.path.isdir(root):
            os.makedirs(root)

        fp = os.path.join(root, *path)
        if os.path.isfile(fp) and unique:
            fp = uniquify(*os.path.splitext(fp))
        return fp

    def search(self, fpath):
        """
        Specific-to-general search in context directories for the named file.
        """
        for base in ('run', 'temp', 'admin', 'context', 'module',
                     'root', 'data', 'results'):
            path = self.path(fpath, base=base)
            if os.path.isfile(path):
                break
        else:
            self.out(fpath, prefix='FileSearchFailed', warning=True)
            return

        return path

    def mkdir(self, *rpath, base=None):
        """
        Create a subdirectory within the run directory.
        """
        dpath = self.path(*rpath, base=base)
        if os.path.isdir(dpath):
            return dpath
        os.makedirs(dpath)
        return dpath

    def subfolder(self, *rpath, prefix=False):
        """
        Make a unique subfolder under the analysis directory.
        """
        stem = self.path(*rpath)
        if prefix:
            path = uniquify(stem, fmt='{:02d}-{stem!s}')
        else:
            path = uniquify(stem)
        subf = rpath[:-1] + (os.path.split(path)[1],)
        return self.mkdir(*subf)

    def filename(self, stem=None, tag=None, ext=None, use_clstag=False,
        use_modname=False, use_step=False, use_runtag=False):
        """
        Create a filename out of many possible elements.
        """
        clstag = step = runtag = None
        if self._tag is not None:
            clstag = self._tag
        if self._lastcall is not None:
            step = self._lastcall['step']
            runtag = self._lastcall['tag']

        fn = []
        if stem:
            fn += [sluggify(stem)]
        if use_clstag and clstag:
            fn += [sluggify(clstag)]
        if use_modname:
            fn += [self._modname]
        if use_step and step:
            fn += [sluggify(step)]
        if use_runtag and runtag:
            fn += [sluggify(runtag)]
        if tag:
            fn += [sluggify(tag)]

        if ext is None:
            ext = ''
        else:
            ext = ext if ext.startswith('.') else f'.{ext}'

        return '+'.join(fn) + ext

    # Console output methods

    def printf(self, *args, **kwargs):
        """Send characters to stdout."""
        self.out.printf(*args, **kwargs)

    def box(self, filled=True, color=None):
        self.out.box(filled=filled, color=color)

    def newline(self):
        self.out.newline()

    def hline(self, color='white'):
        self.out.hline(color=color)

    def launch_anybar(self, color='white'):
        """Create an AnyBar instance for controlling an AnyBar widget."""
        if self._anybar is not None: return
        ab = AnyBar(singleton=True)
        if ab.pid:
            self._anybar = ab
            self.set_anybar_color(color)

    def quit_anybar(self, killall=False):
        """Quit any associated AnyBar instance."""
        if self._anybar is not None:
            self._anybar.quit()
            self._anybar = None
        if killall:
            AnyBar.quit_all()

    def toggle_anybar(self, color1='green', color2='purple'):
        """
        Toggler the AnyBar between two colors as an activity indicator.
        """
        if self._anybar is None: return
        if self._anybar.color not in (color1, color2):
            self.set_anybar_color(color1)
            return
        self.set_anybar_color(
                {color1:color2, color2:color1}[self._anybar.color])

    def set_anybar_color(self, color):
        """If there is an active AnyBar widget, set its color."""
        if self._anybar is None:
            self.launch_anybar(color=color)
            return
        self._anybar.set_color(color)

    # Logging methods

    def open_logfile(self, stem=None, newfile=False, timestamps=False):
        """Start a new log file with optional timestamping."""
        if stem is None:
            if self._running:
                stem = self._lastcall['step']
            else:
                stem = self._name
        fn = self.path(f'{stem}.log')
        self.out.set_timestamps(timestamps)
        self.out.set_outputfile(fn, newfile=newfile)
        return fn

    def close_logfile(self):
        """Close the current log file."""
        self.out.closefile()

    # Step wrapping methods

    def _step_enter(self, method, args, kwargs):
        spec = inspect.getargspec(method)

        # TODO: inspect.getargspec is deprecated since python 3.0. This needs to
        # be rewritten with inspect.getfullargspec or inspect.signature.

        argnames = spec.args[1:]
        argvalues = args[1:]
        params = list(zip(argnames, argvalues))

        tag = kwargs.pop('tag', None)
        if tag is None:
            for name, value in params:
                if name == 'tag':
                    tag = value
                    break

        self._lastcall = info = {
            'time': time.localtime(),
            'subclass': self.__class__.__name__,
            'step': method.__name__,
            'tag': tag,
            'params': params,
            'defaults': spec.defaults,
            'kwname': spec.keywords,  # NOTE: this is just None now...
            'kwvalues': kwargs
        }

        # Run directory is the temp directory during execution
        self._rundir = self._tmpdir
        if not os.path.exists(self._tmpdir):
            os.makedirs(self._tmpdir)

        # Clear out the temp directory for the run
        for fn in os.listdir(self._tmpdir):
            path = os.path.join(self._tmpdir, fn)
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                p = subprocess.run(['rm', '-rf', path])
                if p.returncode != 0:
                    self.out(path, prefix='ProblemRemoving', error=True)

        # Copy the python module file to the run (temp) directory
        pyfile = import_module(self.__class__.__module__).__file__
        p = subprocess.run(['cp', pyfile, self._tmpdir])
        pyfile_copied = p.returncode == 0
        if not pyfile_copied:
            self.out(pyfile, prefix='CopyFailed', warning=True)
        _, basepy = os.path.split(pyfile)

        # Find previous version of python file and generate a diff file
        prev_rundir = os.path.join(self._ctxdir, info['step'])
        if tag: 
            prev_rundir += '+{}'.format(sluggify(tag))
        prev_pyfile = os.path.join(prev_rundir, basepy)
        if pyfile_copied and os.path.isfile(prev_pyfile):
            diffpath = os.path.join(self._tmpdir, '{}.diff'.format(basepy))
            os.system(' '.join(['diff', '-w', prev_pyfile, pyfile,
                    '>"{}"'.format(diffpath)]))

        # Start the AnyBar widget if available
        self.launch_anybar('question')

        # Save any pre-run changes to the key-value context
        self._save_context()

        self.open_logfile(info['step'].replace('_', '-'))
        self.hline()
        self.out('Running step: {}', info['step'], popup=True)
        if params:
            self.out('Call parameters:\n{}',
                     self._format_params(params, spec.defaults))
        if kwargs:
            self.out('Keywords:\n{}', self._format_keywords(kwargs))
        self.hline()

    def _format_params(self, params, defaults, bullet='-'):
        plist = []
        for i, (name, value) in enumerate(reversed(params)):
            if type(value) is pd.DataFrame:
                repr_value = 'DataFrame{}'.format(value.shape)
            else:
                repr_value = repr(value)
            plist.append('{} {} = {}'.format(bullet, name, repr_value))
            if defaults is not None and i < len(defaults):
                dflt = defaults[-1-i]
                if value != dflt:
                    plist[-1] += ' [default: {}]'.format(repr(dflt))
        return '\n'.join(reversed(plist))

    def _format_keywords(self, kwds, bullet='-'):
        return '\n'.join(['{} {} = {}'.format(bullet, k, repr(v))
                          for k,v in kwds.items()])

    def _step_execute(self, method, args, kwargs, status):
        status['OK'] = True
        self._running = True
        self.set_anybar_color('orange')  # to indicate running

        # Save current figure labels and turn off interactive plotting
        prevfigset = frozenset(self._figures.keys())
        plt.ioff()

        try:
            result = method(*args, **kwargs)
        except Exception as e:
            status['OK'] = False
            result = None
            self.out('Exception in {}:\n{}: {}',
                     method.__name__, e.__class__.__name__, str(e),
                     error=True, popup=True, anybar='exclamation')
            pdb.post_mortem(sys.exc_info()[2])
        else:
            self.set_anybar_color('green')
        finally:
            self.close_datafile()
            self.close_datafile(project=True)

            # Show any new figures and restore interactive plotting
            curfigset = frozenset(self._figures.keys())
            plt.ion()
            if prevfigset.symmetric_difference(curfigset):
                plt.show()
            if len(plt.get_figlabels()):
                plt.draw()

        return result

    def _step_exit(self, method, args, kwargs, status):
        step = method.__name__
        tag = self._lastcall['tag']

        if status['OK']:
            self._save_context()
            self._save_call_log()

            # Final output (run) directory is based on method name & tag
            self._rundir = os.path.join(self._ctxdir, step)
            if tag: self._rundir += '+{}'.format(sluggify(tag))
            if not os.path.exists(self._rundir):
                os.makedirs(self._rundir)

            # Move any previous files to a unique 'history' subfolder
            runlist = list(filter(lambda x: x != 'history',
                os.listdir(self._rundir)))
            if len(runlist):
                YMD = time.strftime('%Y-%m-%d', self._lastcall['time'])
                histdir = uniquify(os.path.join(self.mkdir('history'), YMD),
                                    fmt='{stem!s}+{u:02d}')
                os.makedirs(histdir)
                for fn in runlist:
                    runpath = os.path.join(self._rundir, fn)
                    histpath = os.path.join(histdir, fn)
                    os.rename(runpath, histpath)
                self.out(histdir, prefix='FileBackup')

            # Move all the current (temp) output files to the run directory
            # and alias some output files to a special subfolder
            for fn in os.listdir(self._tmpdir):
                tmppath = os.path.join(self._tmpdir, fn)
                runpath = os.path.join(self._rundir, fn)
                os.rename(tmppath, runpath)
                _, ext = os.path.splitext(fn)
                if ext in ('.png', '.pdf', '.mp4'):
                    figdir = self.mkdir('figures', base='module')
                    aliaspath = os.path.join(figdir, fn)
                    if os.path.exists(aliaspath):
                        os.unlink(aliaspath)
                    if os.path.exists(aliaspath + ' alias'):
                        os.unlink(aliaspath + ' alias')
                    if Shell.finder_alias(runpath, figdir) == 0:
                        self.out(aliaspath, prefix='FigureAlias')

            self.out(self._rundir, prefix='OutputDir')
            self._save()

            self.hline()
            dt = time.time() - time.mktime(self._lastcall['time'])
            self.out(f'Finished \'{step}\' in {dt:.1f} seconds.', popup=dt>60)

        self.close_logfile()
        self._running = False
        self.quit_anybar()

        if not status['OK']:
            raise RuntimeError('Stopping due to exception in {}'.format(step))

    def _save_call_log(self):
        logfn = self.path(CALLFILE)
        history = []
        if os.path.isfile(logfn):
            with open(logfn, 'r') as logfd:
                history = logfd.readlines()
        call = self._lastcall
        with open(logfn, 'w') as logfd:
            logfd.write('Login: {}@{}\n'.format(Tenko.user, Tenko.host))
            logfd.write('Time: {}\n'.format(time.strftime('%c', call['time'])))
            signature = ', '.join(['%s=%s' % (k,repr(v))
                for k,v in call['params']])
            if call['kwvalues']:
                if call['params']:
                    signature += ', '
                signature += ', '.join(['%s=%s' % (k,repr(v))
                    for k,v in call['kwvalues'].items()])
            logfd.write('Call: {}.{}({})\n'.format(call['subclass'],
                call['step'], signature))
            logfd.writelines(history)

    def current_step(self):
        """
        Convenience method to return the name of the current step (or None).
        """
        if self._lastcall is None:
            return None
        return self._lastcall['step']

    def current_tag(self):
        """
        Convenience method to return the name of the current run tag (or None).
        """
        if self._lastcall is None:
            return None
        return self._lastcall['tag']

    def dryrun_cleanup(self):
        """
        Clean up the temporary project root directory if this was a dry run.
        """
        if self._dryrun_tempdir:
            self._dryrun_tempdir.cleanup()

    # Parallel methods

    def set_parallel_profile(self, profile):
        """Set the ipython profile to use for the parallel client."""
        if profile:
            parallel.set_default_profile(profile)

    def get_parallel_client(self, profile=None):
        return parallel.client(profile=profile)

    def close_parallel_client(self):
        parallel.close()

    # Datafile methods

    def _set_datafile(self, newpath):
        """Set the path to the context's module-based datafile."""
        if self._datafile:
            self._datafile.close()

        dpath = os.path.abspath(newpath)
        parent, fn = os.path.split(os.path.splitext(dpath)[0])
        self._datafile = DataStore(name=f'{self._modname.title()}Data',
                stem=fn, where=parent)
        self._h5file = self._datafile.path()

    def _set_project_datafile(self, projpath):
        """Set the path to the project's shared datafile."""
        if self._projfile:
            self._projfile.close()

        dpath = os.path.abspath(projpath)
        parent, fn = os.path.split(os.path.splitext(dpath)[0])
        self._projfile = DataStore(name=f'{self._projname.title()}Data',
                stem=fn, where=parent)
        self._h5proj = self._projfile.path()

    def get_datafile(self, readonly=None, *, project=False):
        """Return the context's (default) or project's DataStore object."""
        if project:
            self._projfile.get(readonly=readonly)
            return self._projfile
        self._datafile.get(readonly=readonly)
        return self._datafile

    def flush_datafile(self, *, project=False):
        """Flush the data store file to disk."""
        self.get_datafile(project=project).flush()

    def close_datafile(self, *, project=False):
        """Close the data store file."""
        self.get_datafile(project=project).close()

    def backup_datafile(self, tag=None, *, project=False):
        """Backup the data file and create a clean active copy."""
        self.get_datafile(project=project).backup(tag=tag)

    # Data accessor methods

    def datapath(self, *path, version=None, desc=None, classtag=None,
        step=None, tag=None, root=None):
        """An HDF data path anchored to a versioned & run-tagged root group."""
        if root is not None:
            if isinstance(root, tb.Group):
                root = root._v_pathname
            root = tpath.join('/', root)
            return tpath.join(root, *path)

        # Get step & tag from last run if available
        _last_step = _last_tag = None
        if self._lastcall:
            _last_step = self._lastcall['step']
            _last_tag = self._lastcall['tag']

        # Select given or current values of path components
        Vers = version or self._version
        Desc = desc or self._desc
        Step = step or _last_step
        Ctag = classtag or self._tag
        Rtag = tag or _last_tag

        assert Step is not None, "missing step name for data"

        # Construct the path components
        base = 'v{}'.format(naturalize(Vers))
        if Desc: base += '__{}'.format(naturalize(Desc))
        run = naturalize(Step)
        if Ctag: base += '__tag_{}'.format(naturalize(Ctag))
        if Rtag: run += '__tag_{}'.format(naturalize(Rtag))

        return tpath.join('/', base, run, *path)

    def has_node(self, *path, project=False, **root):
        """Whether a data node exists."""
        p = self.datapath(*path, **root)
        return self.get_datafile(project=project).has_node(p)

    def get_node(self, *path, project=False, **root):
        """Get a handle to a data node if it exists."""
        if len(path) == 1 and isinstance(path[0], tb.Node):
            p = path[0]._v_pathname
        else:
            p = self.datapath(*path, **root)
        return self.get_datafile(project=project).get_node(p)

    def read_node(self, *path, project=False, **root):
        """Read the given node."""
        return self.get_datafile(project=project).get_node(*path,
                    **root).read()

    def read_array(self, *path, project=False, **root):
        """Read array data from the given node."""
        return self.get_datafile(project=project).read_array(
                    self.datapath(*path, **root))

    def read_dataframe(self, *path, project=False, **root):
        """Read pandas dataframe from the given node."""
        return self.get_datafile(project=project).read_dataframe(
                    self.datapath(*path, **root))

    def read_simulation(self, *path, project=False, **root):
        """Read Brian simulation output from the data path."""
        grp = self.get_node(*path, **root)
        if grp._v_attrs['tenko_type'] != 'brian':
            self.out('Not a simulation: {}', grp._v_pathname, error=True)
            raise TypeError('Can only read stored Brian simulation data')

        # While HDF5 file is open, filter parent group for monitor nodes
        network_name = grp._v_attrs['name']
        parent = grp._v_pathname
        dfnames = []
        for mon_node in grp._f_iter_nodes(classname='Group'):
            if 'monitor_type' not in mon_node._v_attrs:
                self.out('skipping {}', mon_node._v_pathname, error=True)
                continue
            name = mon_node._v_name
            dfnames.append(name)

        # Create namedtuple with dataframes loaded from HDF5 file
        dfs = {}
        for name in dfnames:
            dfs[name] = self.read_dataframe(parent, name, project=project)

        simdata = namedtuple('%sData' % network_name, dfs.keys())
        return simdata(**dfs)

    def create_group(self, *path, attrs={}, project=False, **root):
        """Create a new group in the datafile."""
        where, name = tpath.split(self.datapath(*path, **root))
        grp = self.get_datafile(project=project).new_group(where, name)
        if attrs: self._write_v_attrs(grp, attrs)
        self.flush_datafile(project=project)
        return grp

    def create_table(self, descr, *path, attrs={}, project=False, **root):
        """Create a new table in the datafile."""
        return self._new_node('table', self.get_datafile(project=project
                    ).new_table, path, descr, attrs, root)

    def save_array(self, arr, *path, attrs={}, project=False, **root):
        """Save a data array to the datafile."""
        return self._new_node('array', self.get_datafile(project=project
                    ).new_array, path, np.asarray(arr), attrs, root)

    def save_dataframe(self, df, *path, attrs={}, project=False, **root):
        """Save a pandas Series/DataFrame/Panel to the datafile."""
        return self._new_node('dataframe', self.get_datafile(project=project
                    ).new_dataframe, path, df, attrs, root)

    def save_simulation(self, network, *path, attrs={}, project=False, **root):
        """Save Brian simulation output in group/DataFrame structure."""
        import brian2 as br
        monitors = [obj for obj in network.objects if type(obj) in
                        (br.StateMonitor, br.SpikeMonitor,
                            br.PopulationRateMonitor)]

        for mon in monitors:
            mon_attrs = {'name': mon.name,
                'title': f'Network: {network.name}, Monitor: {mon.name}'}
            if hasattr(mon, 'record_variables'):
                record_variables = [v for v in mon.record_variables
                                        if v not in ('t', 'i')]
            columns = {}

            if type(mon) is br.StateMonitor:
                mon_attrs['monitor_type'] = 'state'
                ts = np.tile(mon.t / mon.t.unit, len(mon.record))  # timestamps
                neuron_ix = mon.record.repeat(mon.t.shape[0])  # int unit index

            elif type(mon) is br.SpikeMonitor:
                mon_attrs['monitor_type'] = 'spike'
                ts = mon.t / mon.t.unit  # spike times in seconds
                neuron_ix = (mon.i / mon.i.unit).astype('i')  # int unit index

            elif type(mon) is br.PopulationRateMonitor:
                mon_attrs['monitor_type'] = 'poprate'
                record_variables = ['rate']
                ts = mon.t / mon.t.unit  # timestamp in seconds
                neuron_ix = np.zeros(ts.size) - 1  # flag as population (-1)

            columns.update(t=ts, neuron=neuron_ix)

            for varname in record_variables:
                trace = getattr(mon, varname)

                if type(trace) is np.ndarray:
                    unit = 'scalar'
                elif type(trace) is br.core.variables.VariableView:
                    unit = repr(trace.unit)
                    trace = trace / trace.unit
                elif type(trace) is br.units.fundamentalunits.Quantity:
                    if trace.is_dimensionless:
                        unit = 'scalar'
                    else:
                        unit = repr(trace.dimensions)
                    trace = np.asarray(trace)
                else:
                    self.out('skipping {} \'{}\': unknown type ({})'.format(
                        mon.name, varname, type(trace)), error=True)
                    continue

                if type(mon) is br.StateMonitor:
                    trace = trace.reshape(-1)
                mon_attrs['%s_unit' % varname] = unit
                columns[varname] = trace

            self.save_dataframe(pd.DataFrame(data=columns, columns=['t',
                    'neuron'] + record_variables),
                    *(path + (mon.name,)), attrs=mon_attrs, **root)

        # Write context attributes to parent group of simulation data
        self.get_datafile(project=project)
        grp = self.get_node(*path, project=project, **root)
        grp_attrs = dict(tenko_type='brian', name=network.name)
        grp_attrs = merge_two_dicts(grp_attrs, attrs)
        self._write_v_attrs(grp, grp_attrs)
        return grp

    def _new_node(self, ntype, new_node, path, X, attrs, root, **kwds):
        p = self.datapath(*path, **root)
        where, name = tpath.split(p)
        title = attrs.pop('title', snake2title(name))
        kwds.update(title=title)
        node = new_node(where, name, X, **kwds)
        self._write_v_attrs(node, attrs)
        pathname = node._v_pathname
        if ntype == 'array':
            pathname += ' |{}|'.format('x'.join(list(map(str, X.shape))))
        self.out(f'{pathname} ("{title}")', prefix=f'Saved{ntype.title()}')
        return node

    def _write_v_attrs(self, node, attrs):
        call = self._lastcall
        na = node._v_attrs
        if attrs is not None:
            for k in attrs.keys():
                na[k] = attrs[k]
        if call is None:
            na['time'] = time.localtime()
            return node
        na['time'] = time.mktime(call['time'])
        for key in ('subclass', 'step'):
            na[key] = call[key]
        for name, value in call['params']:
            na[name] = value
        for kwd, value in call['kwvalues'].items():
            if callable(value):
                continue
            if is_specified(value):
                value = value.to_dict()
            na[kwd] = value

    # Figure methods

    def figure_from_template(self, svgpath, label=None, figsize=None, **figkw):
        """Load figure template to create figure and axes.

        Arguments:
        svgpath -- path to SVG file (tiny profile) where rectangles define axes
        label -- optional figure label, defaults to template basename
        figsize -- optionally override the figure size specified in the svg

        Remaining keywords are passed to the `figure` method.

        Returns:
        fig, axd -- tuple of figure handle and axes dictionary
        """
        if not svgpath.endswith('.svg'):
            svgpath += '.svg'
        if label is None:
            label = os.path.basename(os.path.splitext(svgpath)[0])
        self.out('Figure template: {}', svgpath)
        svg = get_svg_figinfo(svgpath)
        figsize = svg['figsize'] if figsize is None else figsize
        figkw.update(figsize=figsize, label=label, clear=True)
        f = self.figure(**figkw)
        axdict = {key: f.add_axes(rect) for key, rect in svg['axes'].items()}
        return f, axdict

    def set_static_figures(self, b=None):
        """Set or toggle whether figure windows are static across calls."""
        old = self._staticfigs

        if b is None:
            self._staticfigs = not self._staticfigs
        else:
            self._staticfigs = bool(b)

        if old != self._staticfigs:
            self._save()

    def figure(self, label=None, clear=False, title=None, handle=None, **kwds):
        """Create or clear a labeled figure window.

        Remaining keywords are passed to `plt.figure`.
        """
        if label is None:
            if self._lastfig:
                label = self._lastfig
            else:
                raise ValueError('no current figure')

        if handle is not None:
            self._figures[label] = handle

        try:
            fig = self._figures[label]
        except KeyError:
            if self._staticfigs:
                kwds.update(num=label)
            fig = self._figures[label] = plt.figure(**kwds)
        else:
            if 'figsize' in kwds:
                fig.set_size_inches(kwds.pop('figsize'), forward=True)
            fig.set(**kwds)
        finally:
            if clear:
                fig.clear()
                fig.subplotpars.left = None
                fig.subplotpars.right = None
                fig.subplotpars.bottom = None
                fig.subplotpars.top = None
                fig.subplotpars.wspace = None
                fig.subplotpars.hspace = None
            self._lastfig = label
            if title is not None:
                fig.suptitle(title)

        return fig

    def savefig(self, label=None, base=None, tag=None, unique=True,
        tight_padding=None, closeafter=False, **savefig):
        """Save an open figure as an image file.

        Arguments:
        label -- optional, label of figure to be saved (default, last figure)
        basepath -- optional, save path for image file up to base name
        tag -- optional, extra text post-pended to file base name
        tight_padding -- optional, padding in inches for tight bounds

        Remaining keywords are passed to `Figure.savefig()`.
        """
        label = label or self._lastfig
        if label not in self._figures:
            self.out('Figure does not exist: {}', label, error=True)
            return

        # Set label-based path if optional base path not specified
        fn = label
        if tag:
            fn += f'+{tag}'

        # Generate unique path with the given format extension
        ext = savefig.pop('format', self._figfmt)
        if ext == 'mpl':
            ext = mpl.rcParams['savefig.format']
        path = self.path(f'{fn}.{ext}', base=base, unique=unique)
        parent, img = os.path.split(path)
        if not os.path.isdir(parent):
            os.makedirs(parent)

        if tight_padding is not None:
            savefig.update(bbox_inches='tight', pad_inches=tight_padding)

        # Persist the current figure save settings
        savefig.update(format=ext)
        self._savefig.update(savefig)

        self._figures[label].savefig(path, **self._savefig)
        self.out(path, prefix='SavedFigure')
        self._savefig_path = path

        if closeafter:
            self.closefig(label=label)

    def openfig(self):
        """Open the last image file saved by using `savefig`."""
        if self._savefig_path is None:
            self.out('There is no previously saved image to open.', error=True)
            return
        if sys.platform != 'darwin':
            self.out('Image viewing only available on OS X.', error=True)
            return
        p = subprocess.run(['open', self._savefig_path])
        if p.returncode != 0:
            self.out('Error opening: {}', self._savefig_path, error=True)
        else:
            self.out('Opened: {}', self._savefig_path)

    def closefig(self, label=None):
        """Close an open figure."""
        if label is None:
            label = self._lastfig

        if label not in self._figures:
            self.out('Figure does not exist: {}', label, error=True)
            return

        plt.close(self._figures[label])
        if not self._holdfigs:
            del self._figures[label]

        if label == self._lastfig:
            self._lastfig = None

    def save_figures(self, **savefig):
        """Save the currently open figures as image files.

        Keyword arguments are passed to `plt.Figure.savefig`:
        """
        for key in list(self._figures.keys()):
            self.savefig(key, **savefig)

    def close_figures(self):
        """Close all open figure windows."""
        self._holdfigs = True
        for key in self._figures:
            self.closefig(key)
        self._holdfigs = False
        self._figures.clear()

    def set_figfmt(self, fmt, save=False):
        """Set the image format for saving figures."""
        assert fmt in ('png','pdf','svg','ps','eps','mpl'), \
                'bad figure format: {}'.format(fmt)
        self._figfmt = fmt
        if save:
            self._save()

    # Movie methods

    def play_movie(self, movie_path=None, scale_dpi=None):
        """
        Play specified movie or the one located at `movie_path` attribute.
        """
        if movie_path is not None:
            movp = movie_path
        elif 'movie_file' in self.c:
            movp = self.path(self.c.movie_file)
        else:
            self.out('Please specify a path to the movie file', error=True)
            return

        if not os.path.isfile(movp):
            self.out(movp, prefix='InvalidFile', error=True)
            return

        self.out(movp, prefix='PlayingMovie')

        dv = subprocess.DEVNULL
        devnull = dict(stdout=dv, stderr=dv)
        p = subprocess.run(['which', 'mpv'], **devnull)
        if p.returncode == 0:
            mpv_cmd = ['mpv', '--loop=yes', '--ontop=yes']
            if scale_dpi is not None:
                wscale = IMACPRO_DPI/scale_dpi
                mpv_cmd.append(f'--window-scale={wscale:.1f}')
            mpv_cmd.append(movp)
            subprocess.run(mpv_cmd, **devnull)
        else:
            self.out('Player \'mpv\' is missing', error=True)