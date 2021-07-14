from config import Cfgs
from run import Execution
import argparse
import os


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='')

    # parser.add_argument('--run', dest='run_mode',
    #                   choices=['train', 'test'],   # !!!
    #                   help='{train, test}',
    #                   type=str, required=True)

    parser.add_argument("--run_mode", type=str, default='test')

    parser.add_argument("--resume", type=bool, default=False)

    parser.add_argument('--log_id', type=int, default=21)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--verbose",type=bool, default=False)

    parser.add_argument('--load_epoch', type=int, default=41)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    arg = Cfgs()

    args = parse_args()
    args_dict = arg.parse_to_dict(args)

    # cfg_file = "cfgs/{}_model.yml".format(args.MODEL)
    # with open(cfg_file, 'r') as f:
    #     yaml_dict = yaml.load(f)

    # args_dict = {**yaml_dict, **args_dict}

    arg.add_args(args_dict)

    print('Hyper Parameters:')
    print(arg)


    execution = Execution(arg)
    execution.run(arg.run_mode)

#conda activate pytorch
