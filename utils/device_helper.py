# coding: utf-8
import torch
import logging
import sys
import random
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def set_device_environment(opt):
    if opt.local_rank == -1 or opt.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not opt.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", opt.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits trainiing: {}".format(
        device, n_gpu, bool(opt.local_rank != -1), opt.fp16))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed)
    opt.n_gpu = n_gpu
    opt.device = device
    return device, n_gpu


def prepare_model(args, model, device, n_gpu):
    """ init my part parameter """

    """ Set device to use """
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model
