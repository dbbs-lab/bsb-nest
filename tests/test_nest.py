import unittest
from unittest.mock import patch

import nest
import numpy as np
from bsb import BootError, CastError, ConfigurationError
from bsb.config import Configuration
from bsb.core import Scaffold
from bsb.services import MPI
from bsb_test import NumpyTestCase, RandomStorageFixture, get_test_config


@unittest.skipIf(MPI.get_size() > 1, "Skipped during parallel testing.")
class TestNest(
    RandomStorageFixture, NumpyTestCase, unittest.TestCase, engine_name="hdf5"
):
    def test_gif_pop_psc_exp(self):
        """Mimics test_gif_pop_psc_exp of NEST's test suite to validate the adapter."""
        pop_size = 500

        cfg = get_test_config("gif_pop_psc_exp")
        sim_cfg = cfg.simulations.test_nest
        sim_cfg.resolution = 0.5
        sim_cfg.cell_models.gif_pop_psc_exp.constants["N"] = pop_size

        network = Scaffold(cfg, self.storage)
        network.compile()

        simulation = None
        vm = None
        nspike = None

        def probe(_, sim, data):
            # Probe and steal some local refs to data that's otherwise encapsulated :)
            nonlocal vm, simulation
            simulation = sim

            # Get the important information out of the sim/data
            cell_m = sim.cell_models.gif_pop_psc_exp
            conn_m = sim.connection_models.gif_pop_psc_exp
            pop = data.populations[cell_m]
            syn = data.connections[conn_m]

            # Add a voltmeter
            vm = nest.Create(
                "voltmeter",
                params={"record_from": ["n_events"], "interval": sim.resolution},
            )
            nest.Connect(vm, pop)

            # Add a spying recorder
            def spy(_):
                nonlocal nspike

                start_time = 1000
                start_step = int(start_time / simulation.resolution)
                nspike = vm.events["n_events"][start_step:]

            data.result.create_recorder(spy)

            # Test node parameter transfer
            for param, value in {
                "V_reset": 0.0,
                "V_T_star": 10.0,
                "E_L": 0.0,
                "Delta_V": 2.0,
                "C_m": 250.0,
                "tau_m": 20.0,
                "t_ref": 4.0,
                "I_e": 500.0,
                "lambda_0": 10.0,
                "tau_syn_in": 2.0,
                "tau_sfa": (500.0,),
                "q_sfa": (1.0,),
            }.items():
                with self.subTest(param=param, value=value):
                    self.assertEqual(value, pop.get(param))

            # Test synapse parameter transfer
            for param, value in (("weight", -6.25), ("delay", 1)):
                with self.subTest(param=param, value=value):
                    self.assertEqual(value, syn.get(param))

        network.simulations.test_nest.post_prepare.append(probe)
        network.run_simulation("test_nest")

        mean_nspike = np.mean(nspike)
        mean_rate = mean_nspike / pop_size / simulation.resolution * 1000.0

        var_nspike = np.var(nspike)
        var_nspike = var_nspike / pop_size / simulation.resolution * 1000.0
        var_rate = var_nspike / pop_size / simulation.resolution * 1000.0

        err_mean = 1.0
        err_var = 6.0
        expected_rate = 22.0
        expected_var = 102.0

        self.assertGreaterEqual(err_mean, abs(mean_rate - expected_rate))
        self.assertGreaterEqual(err_var, var_rate - expected_var)

    @patch("bsb_hdf5.connectivity_set.ConnectivitySet.get_local_chunks")
    def test_empty_conn(self, get_content_mock):
        # Override get_local_chunks to test empty connection sets
        get_content_mock.return_value = []
        cfg = get_test_config("gif_pop_psc_exp")
        cfg.simulations["test_nest"].connection_models["gif_pop_psc_exp"].rule = None
        network = Scaffold(cfg, self.storage)
        network.compile()
        network.run_simulation("test_nest")

    def test_brunel(self):
        cfg = get_test_config("brunel")
        simcfg = cfg.simulations.test_nest

        network = Scaffold(cfg, self.storage)
        network.compile()
        result = network.run_simulation("test_nest")

        spiketrains = result.block.segments[0].spiketrains
        sr_exc, sr_inh = None, None
        for st in spiketrains:
            if st.annotations["device"] == "sr_exc":
                sr_exc = st
            elif st.annotations["device"] == "sr_inh":
                sr_inh = st

        self.assertIsNotNone(sr_exc)
        self.assertIsNotNone(sr_inh)

        rate_ex = len(sr_exc) / simcfg.duration * 1000.0 / sr_exc.annotations["pop_size"]
        rate_in = len(sr_inh) / simcfg.duration * 1000.0 / sr_inh.annotations["pop_size"]

        self.assertAlmostEqual(rate_in, 50, delta=1)
        self.assertAlmostEqual(rate_ex, 50, delta=1)

    def test_brunel_with_conn(self):
        cfg = get_test_config("brunel_wbsb")
        simcfg = cfg.simulations.test_nest

        network = Scaffold(cfg, self.storage)
        network.compile()
        result = network.run_simulation("test_nest")

        spiketrains = result.block.segments[0].spiketrains
        sr_exc, sr_inh = None, None
        for st in spiketrains:
            if st.annotations["device"] == "sr_exc":
                sr_exc = st
            elif st.annotations["device"] == "sr_inh":
                sr_inh = st

        self.assertIsNotNone(sr_exc)
        self.assertIsNotNone(sr_inh)

        rate_ex = len(sr_exc) / simcfg.duration * 1000.0 / sr_exc.annotations["pop_size"]
        rate_in = len(sr_inh) / simcfg.duration * 1000.0 / sr_inh.annotations["pop_size"]

        self.assertAlmostEqual(rate_in, 50, delta=1)
        self.assertAlmostEqual(rate_ex, 50, delta=1)

    def test_iaf_cond_alpha(self):
        """
        Create an iaf_cond_alpha in NEST, and with the BSB, with a base current, and check
        spike times.
        """
        nest.ResetKernel()
        nest.resolution = 0.1
        A = nest.Create("iaf_cond_alpha", 1, params={"I_e": 260.0})
        spikeA = nest.Create("spike_recorder")
        nest.Connect(A, spikeA)
        nest.Simulate(1000.0)

        spike_times_nest = spikeA.get("events")["times"]

        duration = 1000
        resolution = 0.1
        cfg = Configuration(
            {
                "name": "test",
                "storage": {"engine": "hdf5"},
                "network": {"x": 1, "y": 1, "z": 1},
                "partitions": {"B": {"type": "layer", "thickness": 1}},
                "cell_types": {"A": {"spatial": {"radius": 1, "count": 1}}},
                "placement": {
                    "placement_A": {
                        "strategy": "bsb.placement.strategy.FixedPositions",
                        "cell_types": ["A"],
                        "partitions": ["B"],
                        "positions": [[1, 1, 1]],
                    }
                },
                "connectivity": {},
                "after_connectivity": {},
                "simulations": {
                    "test": {
                        "simulator": "nest",
                        "duration": duration,
                        "resolution": resolution,
                        "cell_models": {
                            "A": {
                                "model": "iaf_cond_alpha",
                                "constants": {"I_e": 260.0},
                            }
                        },
                        "connection_models": {},
                        "devices": {
                            "record_A_spikes": {
                                "device": "spike_recorder",
                                "delay": 0.5,
                                "targetting": {
                                    "strategy": "cell_model",
                                    "cell_models": ["A"],
                                },
                            },
                            "voltmeter_A": {
                                "device": "multimeter",
                                "delay": resolution,
                                "properties": ["V_m"],
                                "units": ["mV"],
                                "targetting": {
                                    "strategy": "cell_model",
                                    "cell_models": ["A"],
                                },
                            },
                        },
                    }
                },
            }
        )

        netw = Scaffold(cfg, self.storage)
        netw.compile()
        results = netw.run_simulation("test")
        spike_times_bsb = results.spiketrains[0]
        self.assertTrue(np.unique(spike_times_bsb.annotations["senders"]) == 1)
        membrane_potentials = results.analogsignals[0]
        # last time point is not recorded because of recorder delay.
        self.assertTrue(len(membrane_potentials) == duration / resolution - 1)
        self.assertTrue(np.unique(membrane_potentials.annotations["senders"]) == 1)
        defaults = nest.GetDefaults("iaf_cond_alpha")
        # since current injected is positive, the V_m should be clamped between default
        # initial V_m = -70mV and spike threshold V_th = -55 mV
        self.assertAll(
            (membrane_potentials <= defaults["V_th"])
            * (membrane_potentials >= defaults["V_m"])
        )
        self.assertClose(np.array(spike_times_nest), np.array(spike_times_bsb))

    def test_multimeter_errors(self):
        cfg = get_test_config("gif_pop_psc_exp")
        sim_cfg = cfg.simulations.test_nest
        sim_cfg.devices.update(
            {
                "voltmeter": {
                    "device": "multimeter",
                    "delay": 0.1,
                    "properties": ["V_m", "I_syn"],
                    "units": ["mV"],
                    "targetting": {
                        "strategy": "cell_model",
                        "cell_models": ["gif_pop_psc_exp"],
                    },
                },
            }
        )
        with self.assertRaises(BootError):
            Scaffold(cfg, self.storage)

        sim_cfg.devices.update(
            {
                "voltmeter": {
                    "device": "multimeter",
                    "delay": 0.1,
                    "properties": ["V_m"],
                    "units": ["bla"],
                    "targetting": {
                        "strategy": "cell_model",
                        "cell_models": ["gif_pop_psc_exp"],
                    },
                },
            }
        )
        with self.assertRaises(ConfigurationError):
            Scaffold(cfg, self.storage)

    def test_dc_generator(self):
        duration = 100
        resolution = 0.1
        cfg = Configuration(
            {
                "name": "test",
                "storage": {"engine": "hdf5"},
                "network": {"x": 1, "y": 1, "z": 1},
                "partitions": {"B": {"type": "layer", "thickness": 1}},
                "cell_types": {"A": {"spatial": {"radius": 1, "count": 1}}},
                "placement": {
                    "placement_A": {
                        "strategy": "bsb.placement.strategy.FixedPositions",
                        "cell_types": ["A"],
                        "partitions": ["B"],
                        "positions": [[1, 1, 1]],
                    }
                },
                "connectivity": {},
                "after_connectivity": {},
                "simulations": {
                    "test": {
                        "simulator": "nest",
                        "duration": duration,
                        "resolution": resolution,
                        "cell_models": {
                            "A": {
                                "model": "iaf_cond_alpha",
                                "constants": {
                                    "V_reset": -70,  # V_m, E_L and V_reset are the same
                                },
                            }
                        },
                        "connection_models": {},
                        "devices": {
                            "dc_generator": {
                                "device": "dc_generator",
                                "delay": resolution,
                                "weight": 1.0,
                                "amplitude": 200,  # Low enough so the neuron does not spike
                                "start": 50,
                                "stop": 60,
                                "targetting": {
                                    "strategy": "cell_model",
                                    "cell_models": ["A"],
                                },
                            },
                            "voltmeter_A": {
                                "device": "multimeter",
                                "delay": resolution,
                                "properties": ["V_m"],
                                "units": ["mV"],
                                "targetting": {
                                    "strategy": "cell_model",
                                    "cell_models": ["A"],
                                },
                            },
                        },
                    }
                },
            }
        )

        netw = Scaffold(cfg, self.storage)
        netw.compile()
        results = netw.run_simulation("test")
        v_ms = np.array(results.analogsignals[0])[:, 0]
        self.assertAll(v_ms[: int(50 / resolution) + 1] == -70)
        self.assertAll(
            v_ms[int(50 / resolution) + 1 : int(60 / resolution) + 1] > -70,
            "Current injected should raise membrane potential",
        )

    def test_nest_randomness(self):
        nest.ResetKernel()
        nest.resolution = 0.1
        nest.rng_seed = 1234
        # gif_cond_exp implements a random spiking process.
        # So it's perfect to test the seed
        A = nest.Create(
            "gif_cond_exp",
            1,
            params={"I_e": 200.0, "V_m": nest.random.normal(mean=-70, std=20.0)},
        )
        spikeA = nest.Create("spike_recorder")
        nest.Connect(A, spikeA)
        nest.Simulate(1000.0)
        spike_times_nest = spikeA.get("events")["times"]

        conf = {
            "name": "test",
            "storage": {"engine": "hdf5"},
            "network": {"x": 1, "y": 1, "z": 1},
            "partitions": {"B": {"type": "layer", "thickness": 1}},
            "cell_types": {"A": {"spatial": {"radius": 1, "count": 1}}},
            "placement": {
                "placement_A": {
                    "strategy": "bsb.placement.strategy.FixedPositions",
                    "cell_types": ["A"],
                    "partitions": ["B"],
                    "positions": [[1, 1, 1]],
                }
            },
            "connectivity": {},
            "after_connectivity": {},
            "simulations": {
                "test": {
                    "simulator": "nest",
                    "duration": 1000,
                    "resolution": 0.1,
                    "seed": 1234,
                    "cell_models": {
                        "A": {
                            "model": "gif_cond_exp",
                            "constants": {
                                "I_e": 200.0,
                                "V_m": {
                                    "distribution": "normal",
                                    "mean": -70,
                                    "std": 20.0,
                                },
                            },
                        }
                    },
                    "connection_models": {},
                    "devices": {
                        "record_A_spikes": {
                            "device": "spike_recorder",
                            "delay": 0.5,
                            "targetting": {
                                "strategy": "cell_model",
                                "cell_models": ["A"],
                            },
                        }
                    },
                }
            },
        }
        cfg = Configuration(conf)
        netw = Scaffold(cfg, self.storage)
        netw.compile()
        results = netw.run_simulation("test")
        spike_times_bsb = results.spiketrains[0]
        self.assertClose(np.array(spike_times_nest), np.array(spike_times_bsb))
        self.assertEqual(
            cfg.__tree__()["simulations"]["test"]["cell_models"]["A"]["constants"]["V_m"],
            {
                "distribution": "normal",
                "mean": -70,
                "std": 20.0,
            },
        )

    def test_unknown_distribution(self):
        conf = {
            "name": "test",
            "storage": {"engine": "hdf5"},
            "network": {"x": 1, "y": 1, "z": 1},
            "partitions": {"B": {"type": "layer", "thickness": 1}},
            "cell_types": {"A": {"spatial": {"radius": 1, "count": 1}}},
            "placement": {
                "placement_A": {
                    "strategy": "bsb.placement.strategy.FixedPositions",
                    "cell_types": ["A"],
                    "partitions": ["B"],
                    "positions": [[1, 1, 1]],
                }
            },
            "connectivity": {},
            "after_connectivity": {},
            "simulations": {
                "test": {
                    "simulator": "nest",
                    "duration": 1000,
                    "resolution": 0.1,
                    "cell_models": {
                        "A": {
                            "model": "gif_cond_exp",
                            "constants": {
                                "I_e": 200.0,
                                "V_m": {
                                    "distribution": "bean",
                                    "mean": -70,
                                    "std": 20.0,
                                },
                            },
                        }
                    },
                    "connection_models": {},
                    "devices": {},
                }
            },
        }
        with self.assertRaises(CastError):
            Configuration(conf)
