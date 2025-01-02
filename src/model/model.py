from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

## PyTorch implementation of CDCK2, CDCK5, CDCK6, speaker classifier models
# CDCK2: base model from the paper 'Representation Learning with Contrastive Predictive Coding'
# CDCK5: CDCK2 with a different decoder
# CDCK6: CDCK2 with a shared encoder and double decoders
# SpkClassifier: a simple NN for speaker classification


class CDCK2ForGeoLife(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window,feature_num):  # seq_len=trajectory_window=2048
        super(CDCK2ForGeoLife, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        self.feature_num = feature_num
        
        self.encoder = nn.Sequential( # downsampling factor = 16
            nn.Conv1d(feature_num, 256, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1, bias=False),  # 修改步幅为1
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )


        self.gru = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 17  - self.timestep), size=(1,)).long()  # 下采样因子调整为 16

        z = self.encoder(x)
        z = z.transpose(1, 2)  # [batch_size, sequence_length, input_channels]
        encode_samples = torch.empty((self.timestep, batch, 256), device=x.device)  # [timestep, batch_size, 128]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        #GRU 的机制是输入一个序列（比如 forward_seq），逐步处理序列中的每一个时间步的数据，
        # 同时利用前面时间步的信息（通过隐状态 hidden）进行更新。虽然 GRU 的运算是对每个时间步逐步进行的，
        # 但从代码的角度来看，它是对整个序列 forward_seq 进行了一次性处理。
        output, hidden = self.gru(forward_seq, self.hidden)
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 256)
        pred = torch.empty((self.timestep, batch, 256), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
        
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output

class CDCK2ForAllGeoLife(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window,feature_num):  # seq_len=trajectory_window=2048
        super(CDCK2ForAllGeoLife, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        self.feature_num = feature_num
        
        self.encoder = nn.Sequential( # downsampling factor = 32
            nn.Conv1d(feature_num, 128, kernel_size=10, stride=2, padding=4, bias=False),
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


        self.gru = nn.GRU(128, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(128, 128) for i in range(timestep)])
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 128).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 32  - self.timestep), size=(1,)).long()  # 下采样因子调整为 16

        z = self.encoder(x)
        z = z.transpose(1, 2)  # [batch_size, sequence_length, input_channels]
        print('zzzz',z.shape)
        encode_samples = torch.empty((self.timestep, batch, 128), device=x.device)  # [timestep, batch_size, 128]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 128)
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        #GRU 的机制是输入一个序列（比如 forward_seq），逐步处理序列中的每一个时间步的数据，
        # 同时利用前面时间步的信息（通过隐状态 hidden）进行更新。虽然 GRU 的运算是对每个时间步逐步进行的，
        # 但从代码的角度来看，它是对整个序列 forward_seq 进行了一次性处理。
        output, hidden = self.gru(forward_seq, self.hidden)
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 128)
        pred = torch.empty((self.timestep, batch, 128), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
        
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output


class CDCK6ForGeoLife(nn.Module):
    ''' CDCK2 with double decoder and a shared encoder '''
    def __init__(self, timestep, batch_size, seq_len,feature_num):

        super(CDCK6ForGeoLife, self).__init__()

        self.batch_size = batch_size
        self.trajectory_window = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential( # downsampling factor = 16
            nn.Conv1d(feature_num, 256, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1, bias=False),  # 修改步幅为1
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.gru1 = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk1  = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
        self.gru2 = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk2  = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru1 and gru2
        for layer_p in self.gru1._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru1.__getattr__(p), mode='fan_out', nonlinearity='relu')
        for layer_p in self.gru2._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru2.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)


    def init_hidden1(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def init_hidden2(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def forward(self, x, x_reverse, hidden1, hidden2):
        batch = x.size(0)
        nce = 0
        t_samples = torch.randint(int(self.trajectory_window / 17 - self.timestep), (1,), dtype=torch.long).to(x.device)

        # 第一个 GRU
        z = self.encoder(x)
        z = z.transpose(1, 2)
        encode_samples = torch.zeros((self.timestep, batch, 256), device=x.device)  # 使用 zeros 初始化

        for i in range(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)

        forward_seq = z[:, :t_samples + 1, :]
        output1, hidden1 = self.gru1(forward_seq, hidden1)
        c_t = output1[:, t_samples, :].view(batch, 256)

        pred = torch.zeros((self.timestep, batch, 256), device=x.device)  # 使用 zeros 初始化
        for i in range(self.timestep):
            pred[i] = self.Wk1[i](c_t)

        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], pred[i].transpose(0, 1))
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(batch, device=x.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        # 第二个 GRU
        z = self.encoder(x_reverse)
        z = z.transpose(1, 2)
        encode_samples = torch.zeros((self.timestep, batch, 256), device=x.device)  # 使用 zeros 初始化

        for i in range(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)

        forward_seq = z[:, :t_samples + 1, :]
        output2, hidden2 = self.gru2(forward_seq, hidden2)
        c_t = output2[:, t_samples, :].view(batch, 256)

        pred = torch.zeros((self.timestep, batch, 256), device=x.device)  # 使用 zeros 初始化
        for i in range(self.timestep):
            pred[i] = self.Wk2[i](c_t)

        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], pred[i].transpose(0, 1))
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(batch, device=x.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))

        nce /= -1. * batch * self.timestep
        nce /= 2.
        accuracy = (correct1.item() + correct2.item()) / (batch * 2)

        return encode_samples, pred, hidden1

    def predict(self, x, x_reverse, hidden1, hidden2):
        batch = x.size()[0]

        # first gru
        # input sequence is N*C*L, e.g. 8*1*20480
        z1 = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z1 = z1.transpose(1,2)
        output1, hidden1 = self.gru1(z1, hidden1) # output size e.g. 8*128*256

        # second gru
        z2 = self.encoder(x_reverse)
        z2 = z2.transpose(1,2)
        output2, hidden2 = self.gru2(z2, hidden2)

        return torch.cat((output1, output2), dim=2) # size (64, seq_len, 256)
        #return torch.cat((z1, z2), dim=2) # size (64, seq_len, 512*2)

# 用于新加坡数IOS据集的模型
class CDCK2ForSingapore(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2ForSingapore, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        
        # 更小的步幅以减少下采样的程度
        self.encoder = nn.Sequential( # downsampling factor = 32
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
        
      
        self.gru = nn.GRU(128, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(128, 128) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 128).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 32 - self.timestep), size=(1,)).long()  # 下采样因子调整

        z = self.encoder(x)
        # print('特征提取之后',z.shape) # [batch_size, input_channels, sequence_length]，
        z = z.transpose(1, 2) # [batch_size, sequence_length, input_channels]
  
        encode_samples = torch.empty((self.timestep, batch, 128), device=x.device) # [timestep, batch_size, 512]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 128)

      
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            # 假设：self.hidden 的形状是 [1, 32, 256]（1 层 GRU，32 个样本的批次，256 维隐藏状态）。current_batch_size = 16。
            # 执行 self.hidden[:, :current_batch_size, :] 后，self.hidden 会变成：[1, 16, 256]，表示只保留了前 16 个样本的隐藏状态。
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        output, hidden = self.gru(forward_seq, self.hidden)

       
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 128)

        pred = torch.empty((self.timestep, batch, 128), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
        
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output

class CDCK2ForJakartaAndroid(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2ForJakartaAndroid, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        
        # 更小的步幅以减少下采样的程度
        self.encoder = nn.Sequential(  # downsampling factor = 32
        nn.Conv1d(5, 128, kernel_size=16, stride=2, padding=7, bias=False),  # 卷积核增大
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        
        nn.Conv1d(128, 128, kernel_size=12, stride=2, padding=5, bias=False),  # 卷积核稍微减小
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        nn.Conv1d(128, 128, kernel_size=8, stride=2, padding=3, bias=False),  # 再次减小
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        nn.Conv1d(128, 128, kernel_size=6, stride=2, padding=2, bias=False),  # 继续减小
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),

        nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 最小卷积核
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True)
    )

        
      
        self.gru = nn.GRU(128, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(128, 128) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 128).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 32 - self.timestep), size=(1,)).long()  # 下采样因子调整

        z = self.encoder(x)
        # print('特征提取之后',z.shape) # [batch_size, input_channels, sequence_length]，
        z = z.transpose(1, 2) # [batch_size, sequence_length, input_channels]
  
        encode_samples = torch.empty((self.timestep, batch, 128), device=x.device) # [timestep, batch_size, 512]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 128)

      
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            # 假设：self.hidden 的形状是 [1, 32, 256]（1 层 GRU，32 个样本的批次，256 维隐藏状态）。current_batch_size = 16。
            # 执行 self.hidden[:, :current_batch_size, :] 后，self.hidden 会变成：[1, 16, 256]，表示只保留了前 16 个样本的隐藏状态。
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        output, hidden = self.gru(forward_seq, self.hidden)

       
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 128)

        pred = torch.empty((self.timestep, batch, 128), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
        
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output

# 这个其实没用到
class CDCK2ForJakartaIos(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2ForJakartaIos, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        
        # 更小的步幅以减少下采样的程度
        self.encoder = nn.Sequential( # downsampling factor = 32
            nn.Conv1d(6, 128, kernel_size=10, stride=2, padding=4, bias=False),
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
        
      
        self.gru = nn.GRU(128, 128, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(128, 128) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 128).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 32 - self.timestep), size=(1,)).long()  # 下采样因子调整

        z = self.encoder(x)
        # print('特征提取之后',z.shape) # [batch_size, input_channels, sequence_length]，
        z = z.transpose(1, 2) # [batch_size, sequence_length, input_channels]
  
        encode_samples = torch.empty((self.timestep, batch, 128), device=x.device) # [timestep, batch_size, 512]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 128)

      
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            # 假设：self.hidden 的形状是 [1, 32, 256]（1 层 GRU，32 个样本的批次，256 维隐藏状态）。current_batch_size = 16。
            # 执行 self.hidden[:, :current_batch_size, :] 后，self.hidden 会变成：[1, 16, 256]，表示只保留了前 16 个样本的隐藏状态。
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        output, hidden = self.gru(forward_seq, self.hidden)

       
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 128)

        pred = torch.empty((self.timestep, batch, 128), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
        
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output






class CDCK2ForPorto(nn.Module):
    def __init__(self, timestep, batch_size, seq_len):
        super(CDCK2ForPorto, self).__init__()
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.encoder = nn.Sequential(#Downsampling Factor = 4,适应Porto数据集较少的轨迹点数
            nn.Conv1d(5, 512, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for _ in range(timestep)])
        self.softmax = nn.Softmax(dim=2)
        self.lsoftmax = nn.LogSoftmax(dim=2)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        seq_len = x.size()[-1] // 8  # 动态调整seq_len
        timestep = min(self.timestep, seq_len - 1)  # 动态调整timestep
        
        t_samples = torch.randint(seq_len - timestep, size=(1,)).long()
        
        z = self.encoder(x)
        z = z.transpose(1, 2)

        encode_samples = torch.empty((timestep, batch, 512), device=x.device)
        for i in np.arange(1, timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 512)
        
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
        
        output, hidden = self.gru(forward_seq, self.hidden)
        self.hidden = hidden.detach()
        self.hidden = self.hidden.contiguous()
        c_t = output[:, t_samples, :].view(batch, 256)

        pred = torch.empty((timestep, batch, 512), device=x.device)
        for i in np.arange(0, timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden


    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep

        # 减少下采样，卷积核和步幅都作了调整
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=5, stride=1, padding=2, bias=False),  # 核大小5，步幅1，减少下采样
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1, bias=False),  # 核大小4，步幅1
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),  # 核大小4，步幅2，开始轻微下采样
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 核大小3，步幅2
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 核大小3，步幅2
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )

        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 512) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize GRU weights
        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        t_samples = torch.randint(int(self.trajectory_window / 32 - self.timestep), size=(1,)).long()  # 下采样因子调整

        z = self.encoder(x)
        z = z.transpose(1, 2)

        encode_samples = torch.empty((self.timestep, batch, 512), device=x.device)
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 512)

        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()

        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()

        output, hidden = self.gru(forward_seq, self.hidden)
        self.hidden = hidden.detach().contiguous()
        c_t = output[:, t_samples, :].view(batch, 256)

        pred = torch.empty((self.timestep, batch, 512), device=x.device)
        for i in np.arange(0, self.timestep):
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden

    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output


