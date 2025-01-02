## Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

## Custrom Imports
from src.logger_v1 import setup_logs
from src.data_reader.dataset import RawDatasetGeoLife, RawDatasetSingapore
from src.training_v1 import train,  snapshot
from src.validation_v1 import validation
from src.model.model import CDCK2, CDCK5, CDCK2ForSingapore


############ Control Center and Hyperparameter ###############
run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
# print(run_name)


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128 
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0 
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""
        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

def main():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    
    parser.add_argument('--raw', default='dataset/grab_possi/grab_possi_Jakarta_all_new/ios/jakartaIosForClassifier.hdf5')
    parser.add_argument('--train', default='dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassTrain.txt')
    parser.add_argument('--validation', default='dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassVal.txt')
 

    parser.add_argument('--logging-dir', default='snapshot/jakartaIosForClass/', help='model save directory')
    # log-interval: 这个参数指定了每隔多少个训练批次（batches）后输出一次日志信息。
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train')
    # 用于学习率调度的参数
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')

    # 指定从每个轨迹中采样的窗口长度
    parser.add_argument('--trajectory-window', type=int, default=1024)
    # 指定在处理轨迹数据时，每次跳过的帧数或点数的间隔 如果frame-window设置为1：每个点都进行处理。
    parser.add_argument('--frame-window', type=int, default=1)

    parser.add_argument('--timestep', type=int, default=12) 
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[5], help='List of GPUs to use (e.g., --gpus 0 1)')
    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    # Setting the GPUs to use
    selected_gpus = args.gpus
    print("Using GPUs:", selected_gpus)

    # num_workers: 这个参数指定了数据加载时使用的子进程数量。
    # num_workers=8 表示在数据加载过程中，使用 8 个子进程来并行加载数据。
    params = {'num_workers': 8,
              'pin_memory': False} if use_cuda else {}
           
    # Setting the device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f'cuda:{selected_gpus[0]}')  # Using the first GPU as the main device
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
        if len(selected_gpus) > 1:
            model = nn.DataParallel(model, device_ids=selected_gpus)  # Using multiple GPUs
    else:
        device = torch.device('cpu')
        model = CDCK2ForSingapore(args.timestep, args.batch_size, args.trajectory_window).to(device)
    ## Loading the dataset
    # logger.info('===> loading train, validation and eval dataset')
    
    # 加载数据集
    training_set= RawDatasetSingapore(args.raw,args.train ,args.trajectory_window)
    validation_set = RawDatasetSingapore(args.raw, args.validation,args.trajectory_window)

    # 数据加载器
    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True, **params,drop_last=True) # set shuffle to True
    validation_loader = data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, **params,drop_last=True) # set shuffle to False
    
    # nanxin optimizer  
    optimizer = ScheduledOptim(
        optim.Adam(
            # 过滤掉不需要梯度更新的模型参数。
            filter(lambda p: p.requires_grad, model.parameters()), 
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    # 对所有符合条件的参数张量的元素总数进行求和。
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('### Model summary below###\n {}\n'.format(str(model)))
    logger.info('===> Model total parameter: {}\n'.format(model_params))
    hyperparams_info = "\n".join([f"{key}: {value}" for key, value in vars(args).items()])
    logger.info(f"### Training with the following hyperparameters:\n{hyperparams_info}\n")
    # Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1 
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)
        
        # Save
        if val_acc > best_acc and epoch > 30: 
            best_acc = max(val_acc, best_acc)
            snapshot(args.logging_dir, run_name, {
                'epoch': epoch + 1,
                'validation_acc': val_acc, 
                'state_dict': model.state_dict(),
                'validation_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            })
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1
        
        end_epoch_timer = timer()
        logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))
       
    ## end 
    end_global_timer = timer()
    logger.info("################## Success #########################")
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))



if __name__ == '__main__':
    main()
