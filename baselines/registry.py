from baselines.eadwa import EADWA
from baselines.nmpc import BunkerNMPC

_BUILDERS = {}


def register_controller(type_key, builder):
    _BUILDERS[type_key.upper()] = builder


def is_classical_type(type_key):
    return type_key.upper() in _BUILDERS


def classical_types():
    return sorted(_BUILDERS)


def build_classical_controller(type_key, raw_env, energy_weight):
    return _BUILDERS[type_key.upper()](raw_env, energy_weight)


def _build_nmpc(raw_env, energy_weight):
    prev = raw_env.energy_weight
    raw_env.energy_weight = energy_weight
    try:
        return BunkerNMPC(raw_env)
    finally:
        raw_env.energy_weight = prev


def _build_eadwa(raw_env, energy_weight):
    return EADWA(raw_env, w_e=energy_weight)


register_controller("NMPC", _build_nmpc)
register_controller("EADWA", _build_eadwa)
