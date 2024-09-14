import pickle
import json

def dump_state(name, obj, marshaller='pickle'):
    mode = 'wb+'
    dumper = pickle

    if marshaller == 'json':
        mode = 'w+'
        dumper = json

    with open(name, mode) as f:
        dumper.dump(obj, f)

def load_state(name, marshaller='pickle'):
    mode = 'rb+'
    dumper = pickle

    if marshaller == 'json':
        mode = 'r+'
        dumper = json

    with open(name, mode) as f:
        return dumper.load(f)