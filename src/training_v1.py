import torch
import logging
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

## Get the same logger from main"
logger = logging.getLogger("cdc")

def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data   = data.float().unsqueeze(1).to(device) # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
        optimizer.zero_grad()
        hidden1 = model.init_hidden1(len(data))
        hidden2 = model.init_hidden2(len(data))
        acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))

def train_spk(args, cdc_model, spk_model, device, train_loader, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval() # not training cdc model 
    spk_model.train()
    for batch_idx, [data, target] in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device) # add channel dimension
        target = target.to(device)
        # hidden = cdc_model.init_hidden(len(data))
        
        actual_model = cdc_model.module if isinstance(cdc_model, nn.DataParallel) else cdc_model

        # actual_model.init_hidden(mini_batch_size, use_gpu=True, device=device)
        output = actual_model.predict(data)
        # data = output.contiguous().view((-1,256))
        data = output[:, -1, :]
        target = target.view((-1,1))
        shuffle_indexing = torch.randperm(data.shape[0]) # shuffle frames 
        data = data[shuffle_indexing,:]
        target = target[shuffle_indexing,:].view((-1,))
        optimizer.zero_grad()
        
        actual_model= spk_model.module if isinstance(spk_model, nn.DataParallel) else spk_model
        output = actual_model.forward(data) 
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        acc = 1.*pred.eq(target.view_as(pred)).sum().item()/len(data)
        
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) / frame_window, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, acc, loss.item()))


def train(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_idx, data in enumerate(train_loader):
        # 将数据转换为浮点数，并移动到GPU上
        data = data.float().to(device)  
        
        # 梯度清零
        optimizer.zero_grad()

        # 分配到多个GPU上的批次大小
        mini_batch_size = data.size(0) // len(args.gpus) if args.gpus and len(args.gpus) > 1 else data.size(0)
        if (data.size(0) % len(args.gpus) != 0) and batch_idx > 0:
            mini_batch_size += 1

        # 获取实际模型
        actual_model = model.module if isinstance(model, nn.DataParallel) else model

        # 获取时间步数
        timestep = actual_model.timestep
        
        # 初始化隐藏层
        actual_model.init_hidden(mini_batch_size, use_gpu=True, device=device)


        encode_samples, pred, hidden = model(data)

        nce = 0
        correct = 0
        for i in np.arange(0, timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))#计算当前时间步i中，编码样本encode_samples[i]和预测pred[i]的点积
            correct += torch.sum(torch.eq(torch.argmax(actual_model.softmax(total), dim=0), torch.arange(0, mini_batch_size, device=device)))
            nce += torch.sum(torch.diag(actual_model.lsoftmax(total)))

        nce /= -1.0 * mini_batch_size * timestep
        accuracy = 1.0 * correct.item() / (mini_batch_size * timestep)

        total_loss += nce.item() * data.size(0)
        total_correct += accuracy * data.size(0)
        total_samples += data.size(0)

        nce.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            lr = optimizer.update_learning_rate()
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, avg_acc, avg_loss))


def trainReverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for batch_idx,[data, data_r] in enumerate(train_loader):
        # 将数据转换为浮点数，并移动到GPU上
        data =  data.float().to(device) # add channel dimension
        data_r =data_r.float().to(device) # add channel dimension
        
        # 梯度清零
        optimizer.zero_grad()

        # 分配到多个GPU上的批次大小
        mini_batch_size = data.size(0) // len(args.gpus) if args.gpus and len(args.gpus) > 1 else data.size(0)
        if (data.size(0) % len(args.gpus) != 0) and batch_idx > 0:
            mini_batch_size += 1

        # 获取实际模型
        actual_model = model.module if isinstance(model, nn.DataParallel) else model

        # 获取时间步数
        timestep = actual_model.timestep
        
        # 初始化隐藏层
        hidden1=actual_model.init_hidden1(mini_batch_size, use_gpu=True, device=device)
        hidden2=actual_model.init_hidden2(mini_batch_size, use_gpu=True, device=device)
        
        # 获取预测
        encode_samples, pred, hidden = model(data,data_r, hidden1, hidden2)

        nce = 0
        correct = 0
        for i in np.arange(0, timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))#计算当前时间步i中，编码样本encode_samples[i]和预测pred[i]的点积
            correct += torch.sum(torch.eq(torch.argmax(actual_model.softmax(total), dim=0), torch.arange(0, mini_batch_size, device=device)))
            nce += torch.sum(torch.diag(actual_model.lsoftmax(total)))

        nce /= -1.0 * mini_batch_size * timestep
        accuracy = 1.0 * correct.item() / (mini_batch_size * timestep)

        total_loss += nce.item() * data.size(0)
        total_correct += accuracy * data.size(0)
        total_samples += data.size(0)

        nce.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            lr = optimizer.update_learning_rate()
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), lr, avg_acc, avg_loss))

def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path, run_name + '-model_best.pth')
    
    torch.save(state, snapshot_file)
    logger.info("Snapshot saved to {}\n".format(snapshot_file))
