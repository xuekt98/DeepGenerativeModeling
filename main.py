import argparse
import os
import yaml
import copy
import torch
import random
import numpy as np

from utils import dict2namespace, get_runner
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='BB_base.yml', help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--output_path', type=str, default='output', help="The directory of image outputs")

    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids, 0,1,2,3 cpu=-1')
    parser.add_argument('--MP', action='store_true', default=False, help='use Model Parallel')
    parser.add_argument('--port', type=str, default='12355', help='DDP master port')

    parser.add_argument('--test', action='store_true', default=False, help='Whether to test the model')
    parser.add_argument('--sample_to_calc', action='store_true', default=False, help='Whether to test the model')
    parser.add_argument('--sample_at_start', action='store_true', default=False, help='sample at start')
    parser.add_argument('--save_top', type=int, default=0, help="save top loss checkpoint")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace(config)

    return args, new_config


def set_random_seed(SEED=1234):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def DDP_run_fn(rank, world_size, args, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    set_random_seed(args.seed)

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    config.device = [torch.device("cuda:%d" % local_rank)]
    print('using device:', config.device)
    config.local_rank = local_rank
    runner = get_runner(config.runner, args, config)
    if not args.test:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def CPU_singleGPU_launcher(args, config):
    set_random_seed(args.seed)
    runner = get_runner(config.runner, args, config)
    # runner = eval(args.runner)(args, config)
    if not args.test:
        runner.train()
    else:
        with torch.no_grad():
            runner.test()
    return


def DDP_launcher(world_size, run_fn, args, config):
    mp.spawn(run_fn,
             args=(world_size, copy.deepcopy(args), copy.deepcopy(config)),
             nprocs=world_size,
             join=True)


def main():
    args, config = parse_args_and_config()

    gpu_ids = args.gpu_ids
    config.use_MP = args.MP
    if gpu_ids == "-1": # Use CPU
        config.use_DDP = False
        config.device = [torch.device("cpu")]
        CPU_singleGPU_launcher(args, config)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        gpu_list = gpu_ids.split(",")
        if len(gpu_list) > 1:
            if args.MP:
                config.use_DDP = False
                config.device = []
                for i in range(len(gpu_list)):
                    config.device.append(f'cuda:{gpu_list[i]}')
                CPU_singleGPU_launcher(args, config)
            else:
                config.use_DDP = True
                DDP_launcher(world_size=len(gpu_list), run_fn=DDP_run_fn, args=args, config=config)
        else:
            config.use_DDP = False
            config.device = [torch.device(f"cuda:{gpu_list[0]}")]
            CPU_singleGPU_launcher(args, config)
    return


if __name__ == "__main__":
    main()
