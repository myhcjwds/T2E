"""Explicit configuration for the verified reproduction pipeline."""
import json
import math
from fractions import Fraction
from pathlib import Path


FIXED = {
    'ordinary_window_days': 7,
    'short_history': 100,
    'long_history': 100,
    'fusion_spec': {
        'second_view': {'history_len': 100, 'relation_quota': 100,
                        'weight': 0.5, 'gate': 'always'},
        'prior': {'type': 'backoff', 'half_life': 8760.0, 'pseudocount': 0.1},
        'calibration': 'product',
    },
    'partner_spec': {'pair7': 1.0},
}
VARIABLE = {'model_path', 'physical_witness_days', 'tau', 'packet_cap',
            'max_context_tokens', 'symbolic_weight'}


def packet_cap(value):
    if value.lower() in ('unlimited', 'none', 'null'):
        return None
    number = int(value)
    if number < 1:
        raise ValueError('packet cap must be positive or unlimited')
    return number


def load_config(project, args):
    path = Path(args.config).expanduser() if args.config else project / 'config.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    expected = set(FIXED) | VARIABLE
    if set(config) != expected:
        raise ValueError('Configuration keys differ: ' + str(sorted(set(config) ^ expected)))
    for key, value in FIXED.items():
        if config[key] != value:
            raise ValueError(key + ' is fixed by the verified pipeline; restore config.json')
    for argument, key in (('window', 'physical_witness_days'), ('tau', 'tau'),
                          ('packet_cap', 'packet_cap'), ('symbolic_weight', 'symbolic_weight')):
        value = getattr(args, argument, None)
        if value is not None:
            config[key] = packet_cap(value) if argument == 'packet_cap' else value
    if type(config['physical_witness_days']) is not int or config['physical_witness_days'] < 1:
        raise ValueError('physical_witness_days must be a positive integer')
    ratio = Fraction(str(config['tau']))
    if not 0 < ratio <= 1:
        raise ValueError('tau must be in (0, 1]')
    config['tau'] = str(config['tau'])
    cap = config['packet_cap']
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError('packet_cap must be a positive integer or null')
    weight = config['symbolic_weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('symbolic_weight must be finite and nonnegative')
    if type(config['max_context_tokens']) is not int or config['max_context_tokens'] < 17:
        raise ValueError('max_context_tokens must be an integer greater than 16')
    if not isinstance(config['model_path'], str) or not config['model_path']:
        raise ValueError('model_path must be a nonempty path')
    return config
