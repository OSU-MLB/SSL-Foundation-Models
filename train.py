# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from dotwiz import DotWiz
import logging
import os
import sys
import random
import warnings

import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.core.utils import get_peft_config


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description="Semi-Supervised Learning (USB)")

    """
    Saving & loading of the model.
    """
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="fixmatch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Use tensorboard to plot and save curves",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Use wandb to plot and save curves"
    )
    parser.add_argument(
        "--use_aim", action="store_true", help="Use aim to plot and save curves"
    )

    """
    Training Configuration of FixMatch
    """
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument(
        "--num_train_iter",
        type=int,
        default=20,
        help="total number of training iterations",
    )
    parser.add_argument(
        "--num_warmup_iter", type=int, default=0, help="cosine linear warmup iterations"
    )
    parser.add_argument(
        "--num_eval_iter", type=int, default=10, help="evaluation frequency"
    )
    parser.add_argument("--num_log_iter", type=int, default=5, help="logging frequency")
    parser.add_argument("-nl", "--num_labels", type=int, default=400)
    parser.add_argument("-bsz", "--batch_size", type=int, default=8)
    parser.add_argument(
        "--uratio",
        type=int,
        default=1,
        help="the ratio of unlabeled data to labeled data in each mini-batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="batch size of evaluation data loader (it does not affect the accuracy)",
    )
    parser.add_argument(
        "--ema_m", type=float, default=0.999, help="ema momentum for eval_model"
    )
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    """
    Optimizer configurations
    """
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise learning rate decay, default to 1.0 which means no layer "
        "decay",
    )

    """
    Backbone Net Configurations
    """
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--use_pretrain", default=False, type=str2bool)
    parser.add_argument("--pretrained_path", default="", type=str)

    """
    Algorithms Configurations
    """

    ## core algorithm setting
    parser.add_argument(
        "-alg", "--algorithm", type=str, default="fixmatch", help="ssl algorithm"
    )
    parser.add_argument(
        "--use_cat", type=str2bool, default=True, help="use cat operation in algorithms"
    )
    parser.add_argument(
        "--amp",
        type=str2bool,
        default=False,
        help="use mixed precision training or not",
    )
    parser.add_argument("--clip_grad", type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument(
        "-imb_alg",
        "--imb_algorithm",
        type=str,
        default=None,
        help="imbalance ssl algorithm",
    )

    """
    Data Configurations
    """

    ## standard setting configurations
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="cifar10")
    parser.add_argument("-nc", "--num_classes", type=int, default=10)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--include_lb_to_ulb",
        type=str2bool,
        default="True",
        help="flag of including labeled data into unlabeled data, default to True",
    )

    ## imbalanced setting arguments
    parser.add_argument(
        "--lb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of labeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_imb_ratio",
        type=int,
        default=1,
        help="imbalance ratio of unlabeled data, default to 1",
    )
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
        "number of labels in imbalanced setting",
    )

    ## cv dataset arguments
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)

    ## nlp dataset arguments
    parser.add_argument("--max_length", type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    """
    VTAB
    """
    # parser.add_argument("--evaluate_unsupervised", type=str2bool, default=True)
    parser.add_argument("--evaluate_unsupervised", type=str2bool, default=False)
    parser.add_argument("--train_split", type=str, default="trainval")
    parser.add_argument("--eval_on_test", type=str2bool, default=False, help="Perform extra evaluation on test set")
    parser.add_argument("--eval_on_ulb", type=str2bool, default=True, help="Perform extra evaluation on unlabeled set")
    parser.add_argument("--valid_stage", action="store_true", help="Perform validation stage training")

    """
    multi-GPUs & Distributed Training
    """

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)  # noqa: E501
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="**node rank** for distributed training"
    )
    parser.add_argument(
        "-du",
        "--dist-url",
        default="tcp://127.0.0.1:11111",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    # parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        type=str2bool,
        default=False,
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # early stopping
    # train/total_loss
    parser.add_argument("--es_patience", type=float, default=0, help="Early stopping patience in percentage of total iterations, 0 means no early stopping")
    # parser.add_argument("--es_criteria", type=str, default="train/total_loss_avg")
    parser.add_argument("--resume_es", action="store_true", help="Resume from early stopping")

    parser.add_argument("--train_aug", type=str, default="weak", help="train augmentation")

    parser.add_argument("--no_save", action="store_true", help="Do not save model")

    # PEFT settings
    # peft config
    parser.add_argument("--peft_config", type=dict, default={})
    # vit config
    parser.add_argument("--vit_config", type=dict, default={})

    # PET config
    parser.add_argument("--conf_threshold", type=float, default=0.00)
    parser.add_argument("--conf_ratio", type=float, default=0.00)
    parser.add_argument("--n_rounds", type=int, default=1)
    parser.add_argument("--with_labeled", type=str2bool, default=False)

    # actual training epochs
    parser.add_argument("--actual_epochs", type=int, default=-1)

    # debug
    parser.add_argument("--debug", action="store_true")

    # config file
    parser.add_argument("--c", type=str, default="")
    
    # check config
    parser.add_argument("--check_config", action="store_true")

    # add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(
            argument.name,
            type=argument.type,
            default=argument.default,
            help=argument.help,
        )
    
    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None:
        assert False, "Not supported"
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    ################# Temporary fix #################
    args.gpu = 0
    ################# Temporary fix #################

    # set default peft config
    args.peft_config = get_peft_config(args.peft_config)

    # set save_name if valid_stage is True
    if args.valid_stage:
        print(f"On valid stage")
        print(f"Changing args.save_name from {args.save_name} to {args.save_name}_valid_stage")
        args.save_name = f"{args.save_name}_valid_stage"
        print(f"Changing args.algorithm from {args.algorithm} to fullysupervised")
        # args.algorithm = "fullysupervised"
        args.algorithm = "supervised"
        print(f"Fixing args.lr from {args.lr} to 5e-6 for valid stage")
        args.lr = 1e-5
        print(f"Setting include_lb_to_ulb to False")
        args.include_lb_to_ulb = False
    
    if args.debug:
        ts = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        args.save_name = f"{args.save_name}_debug_{ts}"
        args.load_path = None
        args.resume = False
        # args.num_eval_iter = 2
        print(f"Debugging mode, saving to {args.save_name}, load_path set to None, resume set to False")

    assert args.actual_epochs == -1
    return args


def main(args):
    """
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    """

    assert (
        args.num_train_iter % args.epoch == 0
    ), f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"  # noqa: E501

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume is False:
        import shutil

        shutil.rmtree(save_path)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception("already existing model: {}".format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception("Resume of training requires --load_path in the args")
        if (
            os.path.abspath(save_path) == os.path.abspath(args.load_path)
            and not args.overwrite
        ):
            raise Exception(
                "Saving & Loading paths are same. \
                            If you want over-write, give --overwrite in the argument."
            )

    if args.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu == "None":
        args.gpu = None
    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        assert False, "Not supported"
        print("Distributed training!!!!!!!!!!!!!!!!")
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("Single GPU training!!!!!!!!!!!!!!!!")
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    """
    main_worker is conducted on each GPU.
    """

    global best_acc1
    args.gpu = gpu

    # random seed has to be set for the synchronization of labeled data sampling in each
    # process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    print("########################################################")
    print(args.peft_config)
    print(args.vit_config)
    print("########################################################")
    _net_builder = get_net_builder(args.net, args.net_from_name, peft_config=args.peft_config, vit_config=args.vit_config)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        assert False, "Not supported"
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    if args.valid_stage:
        logger.info("Performing validation stage training")

        resume_load_path = os.path.join(save_path, "latest_model.pth")

        if args.resume and os.path.exists(resume_load_path):
            logger.info(f"Trying to resume from {resume_load_path}")
            try:
                model.load_model(resume_load_path)
            except:
                logger.info(f"Fail to resume load path {resume_load_path}")
                sys.exit(1)
                args.resume = False
        # else:
        #     logger.info(f"Resume load path {resume_load_path} does not exist, performing validation stage training from {args.load_path}")
        #     try:
        #         model.load_checkpoint(args.load_path)
        #     except:
        #         logger.error(f"Fail to load model from {args.load_path}")
        #         sys.exit(1)
        
        model.valid_stage_train()

        for key, item in model.results_dict.items():
            logger.info(f"Model result - {key} : {item}")

    else:
        # If args.resume, load checkpoints from args.load_path
        if args.resume and os.path.exists(args.load_path):
            try:
                model.load_model(args.load_path)
            except:
                logger.info("Fail to resume load path {}".format(args.load_path))
                sys.exit(1)
                args.resume = False
        else:
            logger.info("Resume load path {} does not exist".format(args.load_path))

        if hasattr(model, "warmup"):
            logger.info(("Warmup stage"))
            model.warmup()

        # START TRAINING of FixMatch
        logger.info("Model training")

        if args.check_config:
            sys.exit(0)
        model.train()

        # print validation (and test results)
        for key, item in model.results_dict.items():
            logger.info(f"Model result - {key} : {item}")

        if hasattr(model, "finetune"):
            logger.info("Finetune stage")
            model.finetune()

    logging.warning(f"GPU {args.rank} training is FINISHED")


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)
