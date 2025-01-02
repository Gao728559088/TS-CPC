import argparse
import time
import torch
from src.training_v1 import snapshot
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import numpy as np
import h5py
from torch.utils.data import DataLoader
import time
import os
import logging
from timeit import default_timer as timer
# 必要的导入（例如：模型、数据集类等）
from src.logger_v1 import setup_logs
from src.data_reader.dataset import RawDatasetSingaporeForClass

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

class CPCWithClassification(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window, num_classes=2):
        super(CPCWithClassification, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        self.num_classes = num_classes  # 交通模式的类别数
        
        # 编码器部分，使用卷积层进行特征提取
        self.encoder = nn.Sequential( 
            nn.Conv1d(5, 128, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # 自回归模型部分（GRU）
        self.gru = nn.GRU(128, 128, num_layers=1, batch_first=True)

        # InfoNCE损失计算所需的线性层
        self.Wk = nn.ModuleList([nn.Linear(128, 128) for _ in range(timestep)])

        # 分类输出层
        self.fc_classify = nn.Linear(128, num_classes)
        
    def init_hidden(self, batch_size, device='cuda'):
        return torch.zeros(1, batch_size, 128).to(device)

    def forward(self, x):
        batch = x.size(0)
        
        # 特征提取
        z = self.encoder(x)
        z = z.transpose(1, 2)  # [batch_size, sequence_length, input_channels]

        # GRU计算
        self.hidden = self.init_hidden(batch, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        # CPC的预测：计算预测的潜在特征与实际特征的相似度
        c_t = output[:, -1, :].view(batch, 128)  # 取最后一个时间步的隐藏状态

        # InfoNCE损失计算
        pred = torch.empty((self.timestep, batch, 128), device=x.device)
        for i in range(self.timestep):
            pred[i] = self.Wk[i](c_t)

        # 分类头：使用编码器的输出进行交通模式分类
        classify_out = self.fc_classify(c_t)  # 使用GRU输出的最后一个时刻的特征进行分类

        return pred, classify_out  # 返回用于相似度计算的特征和分类结果

    def predict(self, x):
        batch = x.size(0)
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        # 分类输出
        classify_out = self.fc_classify(output[:, -1, :])  # 使用最后一个时间步的输出进行分类
        return classify_out


class TrafficClassificationModel:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        
        # 初始化模型
        self.model = CPCWithClassification(
            timestep=args.timestep,
            batch_size=args.batch_size,
            trajectory_window=args.trajectory_window,
            num_classes=2
        ).to(self.device)

        # 设置优化器
        self.optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                betas=(0.9, 0.98),
                eps=1e-09,
                weight_decay=1e-4,
                amsgrad=True
            ),
            args.n_warmup_steps
        )

        # 日志设置
        self.run_name = "traffic_classification_" + time.strftime("-%Y-%m-%d_%H_%M_%S")
        self.logger = setup_logs(args.logging_dir, self.run_name)
        self.selected_gpus = args.gpus
        self.logger.info(f"Using GPUs: {self.selected_gpus}")

        # 数据加载
        self.training_set = RawDatasetSingaporeForClass(args.raw, args.train, args.trajectory_window)
        self.validation_set = RawDatasetSingaporeForClass(args.raw, args.validation, args.trajectory_window)
        self.train_loader = DataLoader(self.training_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
        self.validation_loader = DataLoader(self.validation_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        criterion_classification = nn.CrossEntropyLoss()  # 分类损失函数
        criterion_infonce = nn.CrossEntropyLoss()  # InfoNCE损失函数

        for batch_idx, (trajectory_data, labels) in enumerate(self.train_loader):
            trajectory_data, labels = trajectory_data.float().to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 模型输出
            pred, classify_out = self.model(trajectory_data)
            
            # 计算分类损失
            classification_loss = criterion_classification(classify_out, labels)

            # 计算InfoNCE损失
            infonce_loss = 0
            for i in range(pred.size(0)):
                infonce_loss += criterion_infonce(pred[i], labels)

            # 总损失
            total_loss_batch = classification_loss + infonce_loss
            
            total_loss += total_loss_batch.item() * trajectory_data.size(0)
            
            # 计算准确率
            _, predicted = torch.max(classify_out, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_samples += trajectory_data.size(0)

            # 反向传播
            total_loss_batch.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0:
                avg_loss = total_loss / total_samples
                avg_acc = total_correct / total_samples
                print(f"Train Epoch: {epoch} [{batch_idx * len(trajectory_data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)] \tLoss: {avg_loss:.4f} \tAccuracy: {avg_acc:.4f}")

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        criterion_classification = nn.CrossEntropyLoss()  # 分类损失函数
        criterion_infonce = nn.CrossEntropyLoss()  # InfoNCE损失函数
        
        with torch.no_grad():
            for trajectory_data, labels in self.validation_loader:
                trajectory_data, labels = trajectory_data.float().to(self.device), labels.to(self.device)
                
                pred, classify_out = self.model(trajectory_data)
                
                # 计算分类损失
                classification_loss = criterion_classification(classify_out, labels)

                # 计算InfoNCE损失
                infonce_loss = 0
                for i in range(pred.size(0)):
                    infonce_loss += criterion_infonce(pred[i], labels)

                total_loss_batch = classification_loss + infonce_loss
                
                total_loss += total_loss_batch.item() * trajectory_data.size(0)
                _, predicted = torch.max(classify_out, 1)
                correct = (predicted == labels).sum().item()
                total_correct += correct
                total_samples += trajectory_data.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        print(f"Validation Accuracy: {avg_acc:.4f}, Loss: {avg_loss:.6f}")
        return avg_acc, avg_loss

    def fit(self):
        best_acc = 0
        best_loss = np.inf
        best_epoch = -1 

        for epoch in range(1, self.args.epochs + 1):
            epoch_timer = timer()
            
            # 训练和验证
            self.train(epoch)
            val_acc, val_loss = self.validate()

            # 保存最佳模型
            if val_acc > best_acc and epoch > 30: 
                best_acc = val_acc
                self.save_model(epoch, val_acc, val_loss)
                best_epoch = epoch + 1
            elif epoch - best_epoch > 2:
                self.optimizer.increase_delta()
                best_epoch = epoch + 1
            
            # 打印当前epoch的耗时
            end_epoch_timer = timer()
            self.logger.info(f"End of Epoch {epoch}/{self.args.epochs}, Elapsed Time: {end_epoch_timer - epoch_timer}")
        
        # 训练结束
        self.logger.info("Training Complete")

    def save_model(self, epoch, val_acc, val_loss):
        snapshot(self.args.logging_dir, self.run_name, {
            'epoch': epoch + 1,
            'validation_acc': val_acc,
            'state_dict': self.model.state_dict(),
            'validation_loss': val_loss,
            'optimizer': self.optimizer.state_dict(),
        })


def main():
    ## Settings
    parser = argparse.ArgumentParser(description='Traffic Mode Classification with PyTorch')
    
    # 参数设置
    parser.add_argument('--raw', default='dataset/grab_possi/grab_possi_Jakarta_all_new/ios/jakartaIosForClassifier.hdf5')
    parser.add_argument('--train', default='dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassTrain.txt')
    parser.add_argument('--validation', default='dataGenerate/jakartaIosForClassifierGenerate/JakartaIosForClassVal.txt')
    parser.add_argument('--logging-dir', default='snapshot/jakartaIosForClass/', help='model save directory')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='log interval')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',help='number of epochs to train')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--trajectory-window', type=int, default=1024)
    parser.add_argument('--frame-window', type=int, default=1)
    parser.add_argument('--timestep', type=int, default=12) 
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
    parser.add_argument('--gpus', nargs='+', type=int, default=[5], help='List of GPUs to use (e.g., --gpus 0 1)')
    args = parser.parse_args()

    # 初始化模型训练类
    model_trainer = TrafficClassificationModel(args)

    # 训练模型
    model_trainer.fit()

if __name__ == '__main__':
    main()
