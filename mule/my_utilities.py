# import logging
# logger = logging.getLogger(__name__)
# 
# logger.setLevel("INFO")

#
#import tensorflow as tf
#from tensorflow.python import pywrap_tensorflow
import re
#import logging
import yaml as yaml

#warnings.filterwarnings("default")


def load_config_yaml(path_config):
    config = yaml.load(open(path_config, 'r'))
    logging.debug(f"Loaded config file {path_config}")
    return config



def check_versions():
    pass

def list_path():
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    for p in user_paths: print(p) 



def print_tensor_devices():
    
    from tensorflow.python.client import device_lib
    
    
    devices_listing = device_lib.list_local_devices()
    
    devices = list()
    for dev in devices_listing:
        this_dev = str(dev)
        
        dev_dict = dict()
        for item in this_dev.split('\n'):
            if re.search(r':\s',item):
                pair = re.split(r':\s',item)
                dev_dict[pair[0]] = pair[1]
                #print(pair)
                
        devices.append(dev_dict)
    
    for i,dev in enumerate(devices):
        logging.info("Device {}, {}, type {}, memory {}".format(i,
            dev['name'],
            dev['device_type'],
            dev['memory_limit'],            ))