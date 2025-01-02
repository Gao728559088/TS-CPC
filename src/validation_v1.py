import numpy as np
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn

## Get the same logger from main"
logger = logging.getLogger("cdc")

def validationXXreverse(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for [data, data_r] in data_loader:
            data   = data.float().unsqueeze(1).to(device) # add channel dimension
            data_r = data_r.float().unsqueeze(1).to(device) # add channel dimension
            hidden1 = model.init_hidden1(len(data))
            hidden2 = model.init_hidden2(len(data))
            acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)
            total_loss += len(data) * loss 
            total_acc  += len(data) * acc

    total_loss /= len(data_loader.dataset) # average loss
    total_acc  /= len(data_loader.dataset) # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

def validation_spk(args, cdc_model, spk_model, device, data_loader, batch_size, frame_window):
    logger.info("Starting Validation")
    cdc_model.eval() # not training cdc model 
    spk_model.eval()
    total_loss = 0
    total_acc  = 0 

    with torch.no_grad():
        for batch_idx, [data, target] in enumerate(data_loader):
            data = data.float().unsqueeze(1).to(device) # add channel dimension
            target = target.to(device)
            actual_model = cdc_model.module if isinstance(cdc_model, nn.DataParallel) else cdc_model

            output = actual_model.predict(data)
            data = output[:, -1, :]
            # target = target.view((-1,1))
            target = target.view(-1)  # 确保 target 是一维的
            output = spk_model.forward(data) 
            # total_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            total_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            total_acc += pred.eq(target.view_as(pred)).sum().item()

    total_loss /= len(data_loader.dataset)*frame_window # average loss
    total_acc  /= 1.*len(data_loader.dataset)*frame_window # average acc

    logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(
                total_loss, total_acc))

    return total_acc, total_loss

def validation(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            data = data.float().to(device) 

            mini_batch_size = data.size(0) // len(args.gpus) if args.gpus and len(args.gpus) > 1 else data.size(0)
            if (data.size(0) % len(args.gpus) != 0) and batch_idx > 0:
                mini_batch_size += 1

            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            timestep = actual_model.timestep
            actual_model.init_hidden(mini_batch_size, use_gpu=True, device=device) 

            encode_samples, pred, hidden = model(data)

            nce = 0
            correct = 0
            for i in np.arange(0, timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
                correct += torch.sum(torch.eq(torch.argmax(actual_model.softmax(total), dim=0), torch.arange(0, mini_batch_size, device=device)))
                nce += torch.sum(torch.diag(actual_model.lsoftmax(total)))

            nce /= -1.0 * mini_batch_size * timestep
            accuracy = 1.0 * correct.item() / (mini_batch_size * timestep)

            total_loss += nce.item() * data.size(0)
            total_correct += accuracy * data.size(0)
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(avg_loss, avg_acc))

    return avg_acc, avg_loss


def validationReverse(args, model, device, data_loader, batch_size):
    logger.info("Starting Validation")
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx,[data, data_r] in enumerate(data_loader):
            data =  data.float().to(device) # add channel dimension
            data_r =data_r.float().to(device) # add channel dimension
           
            mini_batch_size = data.size(0) // len(args.gpus) if args.gpus and len(args.gpus) > 1 else data.size(0)
            if (data.size(0) % len(args.gpus) != 0) and batch_idx > 0:
                mini_batch_size += 1

            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            timestep = actual_model.timestep

            hidden1=actual_model.init_hidden1(mini_batch_size, use_gpu=True, device=device)
            hidden2=actual_model.init_hidden2(mini_batch_size, use_gpu=True, device=device) 

            encode_samples, pred, hidden = model(data,data_r, hidden1, hidden2)

            nce = 0
            correct = 0
            for i in np.arange(0, timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
                correct += torch.sum(torch.eq(torch.argmax(actual_model.softmax(total), dim=0), torch.arange(0, mini_batch_size, device=device)))
                nce += torch.sum(torch.diag(actual_model.lsoftmax(total)))

            nce /= -1.0 * mini_batch_size * timestep
            accuracy = 1.0 * correct.item() / (mini_batch_size * timestep)

            total_loss += nce.item() * data.size(0)
            total_correct += accuracy * data.size(0)
            total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        logger.info('===> Validation set: Average loss: {:.4f}\tAccuracy: {:.4f}\n'.format(avg_loss, avg_acc))

    return avg_acc, avg_loss
