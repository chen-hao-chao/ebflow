import json
import os
from ebflow.run_lib.train_twodim import run
from ebflow.run_lib.test_twodim import run_test

def main(args):
    # Read the config file
    config_file = os.path.join('ebflow', 'configs', args.dataset, args.loss+'.txt')
    data = open(config_file).read()
    config = json.loads(data)

    if args.eval:
        config['restore'] = args.restore_path
        run_test(config)
    else:
        run(config)