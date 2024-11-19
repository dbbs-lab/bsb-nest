import warnings

import nest
from bsb import DeviceModel, SimulationData, Targetting, config, refs, types

from .adapter import NestAdapter
from .simulation import NestSimulation


@config.node
class NestRule:
    rule = config.attr(type=str, required=True)
    constants = config.catch_all(type=types.any_())
    cell_models = config.reflist(refs.sim_cell_model_ref)


@config.dynamic(attr_name="device", auto_classmap=True, default="external")
class NestDevice(DeviceModel):
    weight = config.attr(type=float, required=True)
    """weight of the connection between the device and its target"""
    delay = config.attr(type=float, required=True)
    """delay of the transmission between the device and its target"""
    targetting = config.attr(
        type=types.or_(Targetting, NestRule), default=dict, call_default=True
    )
    """Targets of the device, which should be either a population or a nest rule"""
    receptor_type = config.attr(type=int, required=False, default=0)
    """Integer ID of the postsynaptic target receptor"""

    def get_dict_targets(
        self,
        adapter: "NestAdapter",
        simulation: "NestSimulation",
        simdata: "SimulationData",
    ) -> dict:
        """
        Get a dictionary from a target group to its NEST Collection
        for each target group of the device.

        :param bsb_nest.NestAdapter adapter:
        :param bsb_nest.NestSimulation simulation: Nest simulation instance
        :param bsb.SimulationData simdata: Simulation data instance
        :return: dictionary of device target group to NEST Collection
        :rtype: dict
        """
        if isinstance(self.targetting, Targetting):
            node_collector = self.targetting.get_targets(adapter, simulation, simdata)
        else:
            node_collector = {
                model: simdata.populations[model][targets]
                for model, targets in simdata.populations.items()
                if not self.targetting.cell_models or model in self.targetting.cell_models
            }
        return node_collector

    @staticmethod
    def _flatten_nodes_ids(dict_targets):
        return sum(dict_targets.values(), start=nest.NodeCollection())

    @staticmethod
    def _invert_targets_dict(dict_targets):
        return {elem: k.name for k, v in dict_targets.items() for elem in v.tolist()}

    def get_target_nodes(
        self,
        adapter: "NestAdapter",
        simulation: "NestSimulation",
        simdata: "SimulationData",
    ):
        """
        Get the NEST Collection of the targets of the device.

        :param bsb_nest.NestAdapter adapter:
        :param bsb_nest.NestSimulation simulation: Nest simulation instance
        :param bsb.SimulationData simdata: Simulation data instance
        :return: Flattened NEST collection with all the targets of the device
        """
        targets_dict = self.get_dict_targets(adapter, simulation, simdata)
        return self._flatten_nodes_ids(targets_dict)

    def connect_to_nodes(self, device, nodes):
        if len(nodes) == 0:
            warnings.warn(f"{self.name} has no targets")
        else:
            try:
                nest.Connect(
                    device,
                    nodes,
                    syn_spec={
                        "weight": self.weight,
                        "delay": self.delay,
                        "receptor_type": self.receptor_type,
                    },
                )

            except Exception as e:
                if "does not send output" not in str(e):
                    raise
                nest.Connect(
                    nodes,
                    device,
                    syn_spec={"weight": self.weight, "delay": self.delay},
                )

    def register_device(self, simdata, device):
        simdata.devices[self] = device
        return device


@config.node
class ExtNestDevice(NestDevice, classmap_entry="external"):
    nest_model = config.attr(type=str, required=True)
    constants = config.dict(type=types.or_(types.number(), str))

    def implement(self, adapter, simulation, simdata):
        simdata.devices[self] = device = nest.Create(
            self.nest_model, params=self.constants
        )
        nodes = self.get_target_nodes(adapter, simulation, simdata)
        self.connect_to_nodes(device, nodes)
