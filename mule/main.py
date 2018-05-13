#!/usr/bin/env python3
"""
THIS IS FOR docopt 
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    manage.py (drive) (--config=<config>) [--model=<model>] [--js] [--offline]
    manage.py (train) (--config=<config>) [--tub=<tub1,tub2,..tubn>]  (--model=<model>) [--no_cache] [--offline]

Options:
    -h --help        Show this screen.
    --tub TUBPATHS   List of paths to tubs. Comma separated. Use quotes to use wildcards. ie "~/tubs/*"
    --js             Use physical joystick.
"""


"""
180501 MJ - Copied from donkey2.py, refactor

"""

#===============================================================================
#--- SETUP Logging
#===============================================================================
import logging.config
import yaml as yaml
import os

# Get the config file
path_logging_conf = os.path.join(os.getcwd(),'logging_config', 'loggingSimpleYaml.yaml')
print(path_logging_conf)
assert os.path.exists(path_logging_conf)
log_config = yaml.load(open(path_logging_conf, 'r'))
logging.config.dictConfig(log_config)

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

logger.debug(f"Logging by {path_logging_conf}")

#===============================================================================
#--- SETUP standard modules
#===============================================================================
import re
import os
from docopt import docopt
import warnings
from pprint import pprint
#===============================================================================
#--- SETUP custom modules
#===============================================================================
import my_utilities as util

with warnings.catch_warnings(): # Suppress warnings!
    warnings.simplefilter("ignore")
    # Disable logging messages from tf - matplotlib (
    #TODO: Better way??
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Main package
    import donkeycar as dk
    
    #import parts
    from donkeycar.parts.camera import PiCamera
    from donkeycar.parts.transform import Lambda
    
    from donkeycar.parts.keras import KerasCategorical
    #logging.getLogger("matplotlib").setLevel(logging.DEBUG)
    
    from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
    from donkeycar.parts.datastore import TubHandler, TubGroup
    from donkeycar.parts.controller import LocalWebController, JoystickController

#===============================================================================
#--- Main script
#===============================================================================


def drive(cfg, model_path=None, use_joystick=False):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    #--- Initialize car
    V = dk.vehicle.Vehicle()
    #print(cfg)
    raise
    cam = PiCamera(resolution=cfg['CAMERA']['CAMERA_RESOLUTION'])
    
    
    V.add(cam, outputs=['cam/image_array'], threaded=True)
    
    if use_joystick or cfg.USE_JOYSTICK_AS_DEFAULT:
        #modify max_throttle closer to 1.0 to have more power
        #modify steering_scale lower than 1.0 to have less responsive steering
        ctr = JoystickController(max_throttle=cfg.JOYSTICK_MAX_THROTTLE,
                                 steering_scale=cfg.JOYSTICK_STEERING_SCALE,
                                 auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)
    else:        
        #This web controller will create a web server that is capable
        #of managing steering, throttle, and modes, and more.
        ctr = LocalWebController()

    
    V.add(ctr, 
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)
    
    #See if we should even run the pilot module. 
    #This is only needed because the part run_condition only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True
        
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])
    
    #Run the pilot if the mode is not user.
    kl = KerasCategorical()
    if model_path:
        kl.load(model_path)
    
    V.add(kl, inputs=['cam/image_array'], 
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')
    
    
    #--- Choose what inputs should change the car.
    def drive_mode(mode, 
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user': 
            return user_angle, user_throttle
        
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        
        else: 
            return pilot_angle, pilot_throttle
        
    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part, 
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'], 
          outputs=['angle', 'throttle'])
    
    
    steering_controller = PCA9685(cfg.STEERING_CHANNEL)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)
    
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)
    
    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])
    
    #--- add tub to save data
    inputs=['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
    types=['image_array', 'float', 'float',  'str']
    
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, run_condition='recording')
    
    #--- run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)
    
    print("You can now go to <your pi ip address>:8887 to drive your car.")


def train(cfg, tub_names, model_name):
    '''
    use the specified data in tub_names to train an artifical neural network
    saves the output trained model as model_name
    '''
    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    def rt(record):
        record['user/angle'] = dk.utils.linear_bin(record['user/angle'])
        return record

    kl = KerasCategorical()
    print('tub_names', tub_names)
    if not tub_names:
        tub_names = os.path.join(cfg.DATA_PATH, '*')
    tubgroup = TubGroup(tub_names)
    train_gen, val_gen = tubgroup.get_train_val_gen(X_keys, y_keys, record_transform=rt,
                                                    batch_size=cfg.BATCH_SIZE,
                                                    train_frac=cfg.TRAIN_TEST_SPLIT)

    model_path = os.path.expanduser(model_name)

    total_records = len(tubgroup.df)
    total_train = int(total_records * cfg.TRAIN_TEST_SPLIT)
    total_val = total_records - total_train
    print('train: %d, validation: %d' % (total_train, total_val))
    steps_per_epoch = total_train // cfg.BATCH_SIZE
    print('steps_per_epoch', steps_per_epoch)

    kl.train(train_gen,
             val_gen,
             saved_model_path=model_path,
             steps=steps_per_epoch,
             train_split=cfg.TRAIN_TEST_SPLIT)

if __name__ == '__main__':
    #--- command line arguments parser
    args = docopt(__doc__)
    print("*** Welcome to Mule DS ***")
    print("Arguments passed:")
    for arg in args:
        if re.match(r"--",arg): # Option
            print("\t{:<10} = {:<10}".format(str(arg),str(args[arg])))
        else: # Command
            print("{:>10} = {:<10}".format(str(arg),str(args[arg])))
    print("**************************")    

    
    #--- Load configuration yaml
    path_config = args['--config']
    assert os.path.exists(path_config), f"Configuration .yml not found at {path_config}"
    cfg = util.load_config_yaml(path_config)
    
    #--- Merge the args and the configuration dictionary into one
    args = {'args':args}
    cfg = {**cfg, **args}
    pprint(cfg)
    if cfg['args']['drive']:
        drive(cfg, model_path = cfg['args']['--model'], use_joystick=cfg['args']['--js'])

    elif cfg['args']['train']:
        tub = cfg['args']['--tub']
        model = cfg['args']['--model']
        cache = not cfg['args']['--no_cache']
        train(cfg, tub, model)





