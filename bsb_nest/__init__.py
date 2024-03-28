"""
NEST simulation adapter for the BSB framework.
"""

from bsb import SimulationBackendPlugin

from . import devices
from .adapter import NestAdapter
from .simulation import NestSimulation

__plugin__ = SimulationBackendPlugin(Simulation=NestSimulation, Adapter=NestAdapter)
__version__ = "0.0.0-b2"
