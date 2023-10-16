import json
import os
from ebflow.run_lib.train_mm import run

def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', 'multimodal10', args.Mtype+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    run(config)