import nest
import quantities as pq
from bsb import ConfigurationError, _util, config, types
from neo import AnalogSignal

from ..device import NestDevice


@config.node
class Multimeter(NestDevice, classmap_entry="multimeter"):
    weight = config.provide(1)
    properties: list[str] = config.attr(type=types.list(str))
    """List of properties to record in the Nest model."""
    units: list[str] = config.attr(type=types.list(str))
    """List of properties' units."""

    def boot(self):
        _util.assert_samelen(self.properties, self.units)
        for i in range(len(self.units)):
            if not self.units[i] in pq.units.__dict__.keys():
                raise ConfigurationError(
                    f"Unit {self.units[i]} not in the list of known units of quantities"
                )

    def implement(self, adapter, simulation, simdata):

        nodes = self.get_target_nodes(adapter, simulation, simdata)
        device = self.register_device(
            simdata,
            nest.Create(
                "multimeter",
                params={
                    "interval": self.simulation.resolution,
                    "record_from": self.properties,
                },
            ),
        )
        self.connect_to_nodes(device, nodes)

        def recorder(segment):
            for prop, unit in zip(self.properties, self.units):
                segment.analogsignals.append(
                    AnalogSignal(
                        device.events[prop],
                        units=pq.units.__dict__[unit],
                        sampling_period=self.simulation.resolution * pq.ms,
                        name=self.name,
                        senders=device.events["senders"],
                    )
                )

        simdata.result.create_recorder(recorder)
