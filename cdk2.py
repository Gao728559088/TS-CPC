from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        
        # 更小的步幅以减少下采样的程度
        self.encoder = nn.Sequential(
            nn.Conv1d(5, 512, kernel_size=10, stride=2, padding=3, bias=False),  # 原步幅5 -> 2
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=8, stride=2, padding=2, bias=False),  # 原步幅4 -> 2
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),  # 原步幅2 -> 1
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),  # 原步幅2 -> 1
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=4, stride=1, padding=1, bias=False),  # 原步幅2 -> 1
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        #   num_layers=1:单GPU,bidirectional=False:单向处理，batch_first=True:要求输入的维度应该是 (batch_size, seq_length, input_size)
        #   输出序列，形状：(batch_size, seq_len, hidden_size)。 对于每个时间步，GRU会生成一个隐状态向量，维度是 hidden_size（在这里是256）。这些隐状态向量在序列中的每个时间步上分别存储，形成一个输出序列。
        #   最终隐状态，形状：(num_layers * num_directions, batch_size, hidden_size)。这是 GRU 在序列的最后一个时间步上生成的隐状态。这个最终隐状态常用于后续的任务，如序列的分类或其他预测。
        #   output 包含了序列中每个时间步的隐状态，这些隐状态提供了对每个时间步的特征表示。
        self.gru = nn.GRU(512, 256, num_layers=1, bidirectional=False, batch_first=True)
        # 创建了一个线性层的列表，其中每个线性层都将负责处理GRU每个时间步的输出。
        # Wk通过一组线性层将每个时间步的256维隐状态重新映射到512维的输出空间
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

    # torch.zeros(1, batch_size, 256) 创建一个全零张量，形状为 (1, batch_size, 256)，用于存储 GRU 的隐状态。
    def init_hidden(self, batch_size, use_gpu=True, device='cuda'):
        self.hidden = torch.zeros(1, batch_size, 256).to(device).contiguous()

    def forward(self, x):
        batch = x.size()[0]
        # 我们处理的是经过编码的特征序列。编码后的序列可能会因为下采样而变得比原始序列短。
        # 为了从这个编码序列中选择一个子序列进行进一步处理（例如，使用 GRU 进行序列建模），我们需要选择一个合适的起始位置。
        # t_samples 生成的随机整数是下采样后编码序列的起始位置。我们用这个位置从编码序列中提取一个子序列。
        t_samples = torch.randint(int(self.trajectory_window / 64 - self.timestep), size=(1,)).long()  # 下采样因子调整

        z = self.encoder(x)
        # print('特征提取之后',z.shape) # [batch_size, input_channels, sequence_length]，
        z = z.transpose(1, 2) # [batch_size, sequence_length, input_channels]

        encode_samples = torch.empty((self.timestep, batch, 512), device=x.device) # [timestep, batch_size, 512]
        for i in np.arange(1, self.timestep + 1):
            # z 的原始形状是 [batch, sequence_length, feature_dim]。 ：的作用（:）的含义是选择所有的批次样本。第三个(:)的作用是选择所有的特征维度。
            # 选择 z[:, t_samples + i, :] 的结果是从 z 中提取在时间步 t_samples + i 上的所有批次的特征。
            # 这操作的结果是一个形状为 [batch, feature_dim] 的张量。
            # 这说明你只选择了时间步 t_samples + i，所以第二维（时间步）被移除，剩下的就是 [batch, feature_dim] 形状的张量。
            # view(batch, 512) 用于调整张量的形状为 [batch, 512]。
            # 在这个上下文中，view() 实际上没有改变数据的结构，因为 z[:, t_samples + i, :] 本身的形状已经是 [batch, 512]。
            # encode_samples[i - 1]：指的是 encode_samples 张量中的第 i-1 行，对应的是提取的第 i 个时间步的特征数据。 
            # encode_samples[i - 1]，也就是对应第i个时间步的所有批次特征数据
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 512)

        # 这一步从编码后的张量 z 中，提取了一个子序列 forward_seq。
        # 该序列包含从第一个时间步到 t_samples + 1 的数据，用作输入给 GRU 层进行处理。也就是以前的信息
        forward_seq = z[:, :t_samples + 1, :]
        # GRU 的隐藏状态，它是一个形状为 [num_layers, batch, hidden_size] 的张量，其中 hidden_size 是 GRU 中的隐藏层维度。
        # .to(x.device)：将隐藏状态 self.hidden 移动到和输入 x 相同的设备
        self.hidden = self.hidden.to(x.device).contiguous()

        # 用于检测当前批次大小，以确保 GRU 的隐藏状态也与之匹配。
        # x 本身的大小没有发生变化。x 被输入到模型的编码器self.encoder(x)中，x.size() 仍然保持不变。
        # 但是，后面的代码仍然需要动态检查批次大小并调整隐藏状态的原因是批次大小可能在运行时发生变化，
        # 例如在不同的训练或推理批次之间。尤其在某些情况下，如使用不固定批次大小（variable batch size）时，
        # 最后一个批次的数据可能少于其他批次，导致批次大小变化。因为在分布式训练的过程中可能最后传入的批次大小不一样
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            # 假设：self.hidden 的形状是 [1, 32, 256]（1 层 GRU，32 个样本的批次，256 维隐藏状态）。current_batch_size = 16。
            # 执行 self.hidden[:, :current_batch_size, :] 后，self.hidden 会变成：[1, 16, 256]，表示只保留了前 16 个样本的隐藏状态。
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()

        # GRU 层的输出 output 是一个形状为 [batch, seq_len, hidden_size] 的张量。
        # GRU 逐步处理输入序列 forward_seq 中的每一个时间步（即时间序列中的每个位置）。
        # 对于每个时间步，GRU 根据当前的输入（某个时间步的特征向量）和之前的隐藏状态，生成当前时间步的输出和更新的隐藏状态。
        # forward_seq：这是输入到 GRU 层的序列，形状是 [batch, sequence_length, feature_dim]，forward_seq 是经过前面卷积层处理后的序列。
        # self.hidden：这是 GRU 层的初始隐藏状态，其形状是 [num_layers, batch, hidden_dim]。这是在 GRU 开始处理新序列时的初始状态。
        # output：形状[batch, sequence_length, hidden_dim]。包含了 GRU 对每个时间步的输出。对于输入序列中的每个时间步，GRU 会计算出一个 256 维的向量（隐藏状态）。
        # 所以 output 记录了每个时间步的 256 维向量，整个序列的所有时间步都会被记录下来。output 中的每个时间步的向量代表了经过 GRU 层处理后的特征，通常这些特征用于后续的预测或进一步的分析。
        # hidden的形状：[num_layers, batch, hidden_dim]。在你的例子中是 [1, batch, 256]。
        # hidden 包含了 GRU 处理完输入序列后的最后隐藏状态。这个状态是 GRU 在序列最后一个时间步的状态，包含了对整个序列的总结信息。
        output, hidden = self.gru(forward_seq, self.hidden)

        # 更新隐藏状态：将当前 GRU 层的最终隐藏状态 hidden 更新到 self.hidden 中，以便于在下一个批次中使用。
        # hidden：这是 GRU 在处理输入序列后得到的最终隐藏状态，形状是 [num_layers, batch, hidden_dim]。
        # detach()，将 hidden 从计算图中分离出来。这样，hidden 的梯度将不会再被计算和传播。这是为了避免在反向传播时对这个隐藏状态的梯度进行计算，因为我们通常不希望在每次迭代时更新之前的隐藏状态的梯度。
        self.hidden = hidden.detach().contiguous()
        # 选择特定时间步：这部分代码从 output 中提取在时间步 t_samples 上的所有批次的输出。结果是一个形状为 [batch, hidden_dim] 的张量，具体来说是 [batch, 256]。
        # view() 方法将张量的形状调整为 [batch, 256]。尽管在这里 output[:, t_samples, :] 的形状已经是 [batch, 256]，
        # output[:, t_samples, :]：从 GRU 的输出 output 中提取在 t_samples 时间步上的所有批次的输出。假设 t_samples 是 25，那么你会从 GRU 输出中提取时间步为 25 的数据。
        # c_t 代表在特定时间步 t_samples 上的所有批次的特征表示。这些特征可以被用于后续的预测或分析。也就是c_t是xt的特征表示，联合cpc的模型图
        # output: 是 GRU 层的整个输出，形状为 [batch_size, sequence_length, hidden_dim]。它包含了每个时间步的隐藏状态，具体说来，是对输入序列每个时间步的特征进行的处理结果。
        # 这里的 c_t 是从 GRU 的输出中提取的，形状为 [batch, 256]。这表示我们从 output 中选择了在时间步 t_samples 上的隐藏状态特
        # GRU 的最终隐藏状态 是 hidden，它代表了整个序列经过 GRU 处理后的总结信息。这个隐藏状态是在序列的最后一个时间步上生成的。
        # c_t：是从 GRU 的输出中提取的在时间步 t_samples 上的隐藏状态，包含了对序列在这一时间点的上下文信息。
        c_t = output[:, t_samples, :].view(batch, 256)

        pred = torch.empty((self.timestep, batch, 512), device=x.device)
        for i in np.arange(0, self.timestep):
            # c_t：这是 GRU 的输出中提取的特定时间步的特征，形状是 [batch, 256]。
            # 这里 self.Wk[i] 是一个线性层，用于将 c_t（时间步 t_samples 的隐藏状态）映射到一个新的特征空间。这样可以用来生成每个时间步的预测 pred[i]。
            pred[i] = self.Wk[i](c_t)

        return encode_samples, pred, hidden
    def predict(self, x):
        batch = x.size()[0]
        z = self.encoder(x)
        z = z.transpose(1, 2)
        self.hidden = self.init_hidden(batch, use_gpu=True, device=x.device)
        output, hidden = self.gru(z, self.hidden)

        return output
    

class CDCK2ForSingapore(nn.Module):
    def __init__(self, timestep, batch_size, trajectory_window):  # seq_len=trajectory_window=2048
        super(CDCK2ForSingapore, self).__init__()
        self.batch_size = batch_size
        self.trajectory_window = trajectory_window
        self.timestep = timestep
        
        # 更小的步幅以减少下采样的程度
        self.encoder = nn.Sequential( # downsampling factor = 32
            nn.Conv1d(5, 256, kernel_size=10, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=8, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
         )
        
      
        self.gru = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(256, 256) for i in range(timestep)])
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
        # print('特征提取之后',z.shape) # [batch_size, input_channels, sequence_length]，
        z = z.transpose(1, 2) # [batch_size, sequence_length, input_channels]
  
        encode_samples = torch.empty((self.timestep, batch, 256), device=x.device) # [timestep, batch_size, 512]
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, 256)

      
        forward_seq = z[:, :t_samples + 1, :]
        self.hidden = self.hidden.to(x.device).contiguous()
        current_batch_size = x.size(0)
        if current_batch_size != self.hidden.size(1):
            # 假设：self.hidden 的形状是 [1, 32, 256]（1 层 GRU，32 个样本的批次，256 维隐藏状态）。current_batch_size = 16。
            # 执行 self.hidden[:, :current_batch_size, :] 后，self.hidden 会变成：[1, 16, 256]，表示只保留了前 16 个样本的隐藏状态。
            self.hidden = self.hidden[:, :current_batch_size, :].contiguous()
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
