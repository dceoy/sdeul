#!/usr/bin/env python

from importlib.metadata import version

if __package__:
    __version__ = version(__package__)
else:
    raise ImportError("Can't determine version number")
