import os.path
import time
import re
import json
import numpy as np
import collections
import collections.abc
import random
from parts.base import BasePart


# TODO: filter on input_keys for WriteStore
# TODO: add handler for multiple read/write stores
# TODO: document properly
# TODO: multithread, multiprocess or asyncio these ops


def _json_load(path):
    with open(path, 'r') as fd:
         return json.load(fd)

def _none_load(path):
    return None


_LOADER_LOOKUP = {'npy': np.load, 'json': _json_load}
_EXTENSION_LOOKUP = {np.ndarray: 'npy', collections.abc.Mapping: 'json'}



class ReadStore(BasePart):
    ''' Reads data from storage path into a state dictionary

        Expects filenames in a certain format:

        <state-key>_<timestamp-in-seconds>.<extension>
    '''
    input_keys = ()
    output_keys = ()

    def __init__(self, path, output_keys=None, shuffled=False):
        self.path = os.path.realpath(path)
        self.output_keys = output_keys or ()
        self.shuffled = shuffled

        # TODO: remove to external verification
        if not os.path.exists(self.path):
            msg = '{} does not exist'.format(self.path)
            raise FileNotFoundError()

        self.data_specifiers = self._retrieve_data_specifiers()


    def start(self):
        if self.shuffled:
            self.random_generator = random.Random()
            self.random_generator.seed(0)


    def transform(self, state):
        old_state = state
        try:
            state = self._generate_state()
        except StopIteration:
            state = old_state
            raise StopIteration

    def stop(self):
        pass


    def _retrieve_data_specifiers(self):
        data_specifiers = {}
        for element in os.listdir(self.path):
            match = re.fullmatch('(\w+)_([0-9]{9}).(.*)', element)
            if match:
                filename = os.path.join(self.path, element)

                datum_key, timestamp, extension = match.groups()

                timestamp = int(timestamp)

                datum_specifier = (datum_key, extension, filename)
                if data_specifiers.get(timestamp):
                    data_specifiers[timestamp].append(datum_specifier)
                else:
                    data_specifiers[timestamp] = [datum_specifier]

        return collections.OrderedDict(sorted(data_specifiers.items(), key=lambda d: d[0]))


    def _generate_state(self):
        ''' Generates state dictionary from files 

            Expects data to be either a Mapping whose keys are state keys'''

        data_specifier_keys = list(self.data_specifiers)

        if self.shuffled:
            self.random_generator.shuffle(data_specifier_keys)

        for key in data_specifier_keys:
            state = {}
            for key, extension, filename in self.data_specifiers[key]:
                # open file
                data = _LOADER_LOOKUP.get(extension, _none_load)(filename)

                if isinstance(data, collections.abc.Mapping)
                    if self.output_keys:
                        data = {k:v for k, v in data.items() if k in self.output_keys} 
                    # TODO: requires python >= 3.5, make this 3.4 compatible
                    state = {**state, **data}
                else if key in self.output_keys:
                    state[key] = data 

            yield state



class WriteStore(BasePart):
    ''' Writes state dictionary to storage path

        Writes filenames in a certain format:

        <state-key>_<timestamp-in-seconds>.<extension>
    '''
    input_keys = ()
    output_keys = ()

    def __init__(self, path, input_keys=None):
        self.path = os.path.realpath(path)
        self.input_keys = input_keys or ()

        # TODO: remove to external verification
        if not os.path.exists(self.path):
            os.makedirs(self.path)


    def start(self):
        pass


    def transform(self, state):
        timestamp = time.time()

        local_state = {}

        for key, value in state.items():
            if isinstance(value, np.ndarray):
                filename = '{}_{}.{}'.format(key, timestamp, _EXTENSION_LOOKUP[np.ndarray])
                np.save(filename, value)

            else:
                local_state[key] = value

        filename = 'state_{}.{}'.format(timestamp, _EXTENSION_LOOKUP[collections.abc.Mapping])
        with open(filename, 'w') as fd:
            json.dump(local_state)


    def stop(self):
        pass
