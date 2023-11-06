from neo import SpikeTrain
from ..device import NestDevice
from bsb import config


@config.node
class PoissonGenerator(NestDevice, classmap_entry="poisson_generator"):
    rate = config.attr(type=float, required=True)

    def implement(self, adapter, simulation, simdata):
        import bsb_nest

        nodes = self.get_target_nodes(adapter, simulation, simdata)
        device = self.register_device(
            simdata, bsb_nest.Create("poisson_generator", params={"rate": self.rate})
        )
        sr = bsb_nest.Create("spike_recorder")
        bsb_nest.Connect(device, sr)
        self.connect_to_nodes(device, nodes)

        def recorder(segment):
            segment.spiketrains.append(
                SpikeTrain(
                    sr.events["times"],
                    units="ms",
                    senders=sr.events["senders"],
                    t_stop=simulation.duration,
                    device=self.name,
                )
            )

        simdata.result.create_recorder(recorder)
