"""
NEST simulation adapter for the BSB framework.
"""

from bsb.simulation import SimulationBackendPlugin
from .adapter import NestAdapter
from .simulation import NestSimulation
from . import devices


__plugin__ = SimulationBackendPlugin(Simulation=NestSimulation, Adapter=NestAdapter)
__version__ = "0.0.0b1"