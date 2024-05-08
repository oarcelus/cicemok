import mph
from typing import Optional
import numpy as np
from cicemok.configuration import ComsolConfiguration


def start_client(cores: Optional[int] = None) -> mph.Client:
    if cores is None:
        client: mph.Client = mph.start()
    else:
        client: mph.Client = mph.start(cores=cores)

    return client


def load_model(client: mph.Client, filename: str) -> mph.Model:
    model: mph.Model = client.load(filename)

    return model


def edit_model_parameters(model: mph.Model, params: dict[str, tuple[float, str]]) -> mph.Model:
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

    dt = config.experiment.texp / config.experiment.experiment.shape[0]
    t = 0.0
    model.java.physics().remove("ev")
    model.java.physics().create("ev", "Events", "geom1")
    model.java.component("comp1").physics("ev").create("ds1", "DiscreteStates", -1)
    model.java.component("comp1").physics("ev").feature("ds1").setIndex("dim", config.iappname, 0, 0)
    model.java.component("comp1").physics("ev").feature("ds1").setIndex("dimInit", f"{config.i1Cname}*{config.experiment.experiment[0]}", 0, 0)
    for i, step in enumerate(config.experiment.experiment):
        model.java.component("comp1").physics("ev").create(f"expl{i}", "ExplicitEvent", -1)
        model.java.component("comp1").physics("ev").feature(f"expl{i}").set("start", t)
        model.java.component("comp1").physics("ev").feature(f"expl{i}").setIndex("reInitName", config.iappname, 0, 0)
        model.java.component("comp1").physics("ev").feature(f"expl{i}").setIndex("reInitValue", f"{config.i1Cname}*{step}", 0, 0)
        t += dt

    return model


def set_timesteps(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    dt = config.experiment.texp / config.experiment.experiment.shape[0] / 10

    model.java.study("std1").feature("time").set("tunit", "s")
    model.java.study("std1").feature("time").set("tlist", f"range(0,{dt},{config.experiment.texp})")

    return model


def set_soc(model: mph.Model, config: ComsolConfiguration) -> mph.Model:
    assert config.experiment is not None
    assert config.experiment.experiment is not None

    model.java.component("comp1").variable("var1").set(config.isocname, config.experiment.isoc)

    return model


def run_comsol_model(input: np.ndarray, model: mph.Model, config: ComsolConfiguration) -> np.ndarray | None:
    parameters: dict[str, tuple[float, str]] = {k: (v, u) for k, u, v in zip(config.names, config.unit, input.tolist())}
    model = edit_model_parameters(model, parameters)
    try:
        model.solve()
        result: np.ndarray = model.evaluate(config.expression, dataset=config.database)
        return result
    except Exception:
        return None

