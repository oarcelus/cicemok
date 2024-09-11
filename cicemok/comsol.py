import multiprocessing
from typing import Optional

import matplotlib.pyplot as plt
import mph
import numpy as np

from cicemok.configuration import ComsolConfiguration


def start_client(cores: Optional[int] = None) -> mph.Client:
    mph.option(name="session", value="stand-alone")
    if cores is None:
        client: mph.Client = mph.start()
    else:
        client: mph.Client = mph.start(cores=cores)

    return client


def load_model(client: mph.Client, filename: str) -> mph.Model:
    model: mph.Model = client.load(filename)

    return model


def edit_model_parameters(
    model: mph.Model, params: dict[str, tuple[float, str]]
) -> mph.Model:
    for key, value in params.items():
        s = model.parameter(key)
        assert isinstance(s, str)

        val = value[0]
        unit = value[1]
        newparam: str = f"{val}[{unit}]"
        model.parameter(key, newparam)

    return model


def set_configuration(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    model = set_events(model, config)
    model = set_timesteps(model, config)
    model = set_soc(model, config)

    return model


def set_events(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    model.java.physics().remove("ev")
    model.java.physics().create("ev", "Events", "geom1")
    model.java.component("comp1").physics("ev").create("ds1", "DiscreteStates", -1)
    model.java.component("comp1").physics("ev").feature("ds1").setIndex(
        "dim", config.evname, 0, 0
    )
    model.java.component("comp1").physics("ev").feature("ds1").setIndex(
        "dimInit", f"{config.experiment.experiment[0, 1]}", 0, 0
    )
    for i, step in enumerate(config.experiment.experiment):
        model.java.component("comp1").physics("ev").create(
            f"expl{i}", "ExplicitEvent", -1
        )
        model.java.component("comp1").physics("ev").feature(f"expl{i}").set("start", step[0])
        model.java.component("comp1").physics("ev").feature(f"expl{i}").setIndex(
            "reInitName", config.evname, 0, 0
        )
        model.java.component("comp1").physics("ev").feature(f"expl{i}").setIndex(
            "reInitValue", f"{step[1]}", 0, 0
        )

    return model


def set_timesteps(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    dt = config.experiment.texp / config.experiment.experiment.shape[0] / 20

    model.java.study("std1").feature("time").set("tunit", "s")
    model.java.study("std1").feature("time").set(
        "tlist", f"range(0,{dt},{config.experiment.texp})"
    )

    return model


def set_soc(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    model.java.component("comp1").variable("var1").set(
        config.isocname, config.experiment.isoc
    )

    return model


def set_model_parameters(
    input: np.ndarray, model: mph.Model, config: ComsolConfiguration
) -> mph.Model:
    parameters: dict[str, tuple[float, str]] = {
        k: (v, u) for k, u, v in zip(config.names, config.unit, input.tolist())
    }
    model = edit_model_parameters(model, parameters)

    return model


def run_comsol_model(
    input: np.ndarray, model: mph.Model, config: ComsolConfiguration
) -> np.ndarray | None:

    model = set_model_parameters(input, model, config)
    try:
        model.solve()
        result: np.ndarray = model.evaluate(config.expression, dataset=config.database)

        return result
    except Exception:
        return None


def setup_comsol_worker(
    ncores: int, config: ComsolConfiguration, event: multiprocessing.Event
):
    global model

    client = start_client(cores=ncores)
    model = client.load(config.filename)
    event.set()


def set_model_parameters_pool(input: np.ndarray, config: ComsolConfiguration):
    global model

    model = set_model_parameters(input, model, config)


def comsol_worker_pool(sample: np.ndarray, config: ComsolConfiguration):
    global model

    model = set_configuration(model, config)
    result = run_comsol_model(sample, model, config)

    return result
