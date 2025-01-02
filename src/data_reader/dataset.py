import numpy as np
import torch
from torch.utils import data
import h5py
from scipy.io import wavfile
from collections import defaultdict
from random import randint

class ForwardLibriSpeechRawXXreverseDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        original = self.h5f[utt_id][:]

        return utt_id, self.h5f[utt_id][:], original[::-1].copy()
 
class ForwardLibriSpeechReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        original = self.h5f[utt_id][:]

        return utt_id, original[::-1].copy() # reverse
    
class ForwardLibriSpeechRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 

        return utt_id, self.h5f[utt_id][:]
    
class ReverseRawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ RawDataset trained reverse;
            raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index+self.audio_window]
        return original[::-1].copy() # reverse 

class ForwardDatasetSITWSilence(data.Dataset):
    ''' dataset for forward passing sitw without vad '''
    def __init__(self, wav_file):
        """ wav_file: /export/c01/jlai/thesis/data/sitw_dev_enroll/wav.scp
        """
        self.wav_file  = wav_file

        with open(wav_file) as f:
            temp = f.readlines()
        self.utts = [x.strip().split(' ')[0] for x in temp]
        self.wavs = [x.strip().split(' ')[1] for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        wav_path = self.wavs[index] # get the wav file path 
        fs, data = wavfile.read(wav_path)

        return self.utts[index], data

class ForwardDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for forward passing swbd_sre or sre16 without vad '''
    def __init__(self, wav_dir, scp_file):
        """ wav_dir: /export/c01/jlai/thesis/data/swbd_sre_combined/wav/
            list_file: /export/c01/jlai/thesis/data/swbd_sre_combined/list/log/swbd_sre_utt.{1..50}.scp
        """
        self.wav_dir  = wav_dir

        with open(scp_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        path   = self.wav_dir + utt_id
        fs, data = wavfile.read(path)

        return utt_id, data

class RawDatasetSwbdSreOne(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''
    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training3.txt, list/val3.txt
        """
        self.raw_file  = raw_file 

        with open(list_file) as f:
            temp = f.readlines()
        all_utt = [x.strip() for x in temp]
    
        # dictionary mapping unique utt id to its number of voied segments
        self.utts = defaultdict(lambda: 0)
        for i in all_utt: 
            count  = i.split('-')[-1]
            utt_uniq = i[:-(len(count)+1)]
            self.utts[utt_uniq] += 1 # count 

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts.keys()[index] # get the utterance id 
        count  = self.utts[utt_id] # number of voiced segments for the utterance id  
        select = randint(1, count)
        h5f = h5py.File(self.raw_file, 'r')
        
        return h5f[utt_id+'-'+str(select)][:]

class RawDatasetSwbdSreSilence(data.Dataset):
    ''' dataset for swbd_sre without vad; for training cpc with ONE voiced/unvoiced segment per recording '''
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training2.txt, list/val2.txt
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        h5f = h5py.File(self.raw_file, 'r')
        utt_len = h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 

        return h5f[utt_id][index:index+self.audio_window]

class RawDatasetSwbdSre(data.Dataset):
    ''' dataset for swbd_sre with vad ; for training cpc with ONE voiced segment per recording '''
    def __init__(self, raw_file, list_file):
        """ raw_file: swbd_sre_combined_20k_20480.h5
            list_file: list/training.txt
        """
        self.raw_file  = raw_file 

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        h5f = h5py.File(self.raw_file, 'r')

        return h5f[utt_id][:]

class RawDatasetSpkClass(data.Dataset):
    def __init__(self, raw_file, list_file, index_file, audio_window, frame_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            index_file: spk2idx
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.frame_window = frame_window

        with open(list_file) as f:
            temp = f.readlines()
        self.utts = [x.strip() for x in temp]

        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = int(i.split(' ')[1])
            self.spk2idx[spk] = idx

    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):#haofq20240121
        utt_id = self.utts[index]  # get the utterance id
        h5f = h5py.File(self.raw_file, 'r')
        utt_len = h5f[utt_id].shape[0]  # get the number of data points in the utterance

        if utt_len < self.audio_window:
            # 如果音频片段长度小于窗口大小，则填充静音
            padding_size = self.audio_window - utt_len
            audio_data = h5f[utt_id][:]
            padded_audio_data = np.pad(audio_data, (0, padding_size), 'constant', constant_values=(0, 0))
        else:
            # 随机选择音频片段中的一个窗口
            start_index = np.random.randint(utt_len - self.audio_window + 1)
            padded_audio_data = h5f[utt_id][start_index:start_index + self.audio_window]

        speaker = utt_id.split('-')[0]
        label = torch.tensor(self.spk2idx[speaker])

        return padded_audio_data, label.repeat(self.frame_window)


# class RawDatasetGeoLife(data.Dataset):
#     def __init__(self, raw_file, list_file, trajectory_window):
#         """ raw_file: geoLife.hdf5
#             list_file: list/training.txt
#             trajectory_window: 20480 (假设这是每次读取轨迹点的数量)
#         """
#         self.raw_file = raw_file 
#         self.trajectory_window = trajectory_window
#         self.trajectories = []

#         with open(list_file) as f:
#             temp = f.readlines()
#         temp = [x.strip() for x in temp]#remove the space in the head and tail of the string
        
#         self.h5f = h5py.File(self.raw_file, 'r')
#         for i in temp:  # sanity check
#             traj_len = self.h5f[i].shape[0]
#             if traj_len > self.trajectory_window:
#                 self.trajectories.append(i)

#     def __len__(self):
#         """Denotes the total number of trajectories"""
#         return len(self.trajectories)

#     def __getitem__(self, index):
#         traj_id = self.trajectories[index]  # get the trajectory id 
#         traj_len = self.h5f[traj_id].shape[0]  # get the number of data points in the trajectory
#         start_index = np.random.randint(traj_len - self.trajectory_window + 1)  # get the index to read part of the trajectory into memory 

#         trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
#         # Assuming trajectory_data format: 'Latitude', 'Longitude', 'Zero', 'Altitude', 'Days'
#         # You can process trajectory_data here as needed
#         trajectory_data = np.transpose(trajectory_data)

#         return torch.tensor(trajectory_data, dtype=torch.float)

class RawDatasetPorto(data.Dataset):
    def __init__(self, raw_file, list_file, trajectory_window):
        """ raw_file: portoTrain.hdf5
            list_file: list/training.txt
            trajectory_window: 30 (假设这是每次读取轨迹点的数量)
        """
        self.raw_file = raw_file 
        self.trajectory_window = trajectory_window
        self.trajectories = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]  # remove the space in the head and tail of the string
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:
            traj_len = self.h5f[i].shape[0]
            self.trajectories.append(i)  # 添加所有轨迹ID，不再进行长度过滤

    def __len__(self):
        """Denotes the total number of trajectories"""
        return len(self.trajectories)

    def __getitem__(self, index):
        traj_id = self.trajectories[index]  # get the trajectory id 
        traj_len = self.h5f[traj_id].shape[0]  # get the number of data points in the trajectory
        
        if traj_len >= self.trajectory_window:
            start_index = np.random.randint(traj_len - self.trajectory_window + 1)  # get the index to read part of the trajectory into memory
            trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
        else:
            # 对于长度小于trajectory_window的轨迹，进行填充或其他处理
            padding = np.zeros((self.trajectory_window - traj_len, self.h5f[traj_id].shape[1]))
            trajectory_data = np.concatenate((self.h5f[traj_id][:], padding), axis=0)
        
        trajectory_data = np.transpose(trajectory_data)  # Assuming 'Latitude', 'Longitude', 'Zero', 'Altitude', 'Days'

        return torch.tensor(trajectory_data, dtype=torch.float)


 
class RawDatasetGeoLife(data.Dataset):
    def __init__(self, raw_file, list_file, trajectory_window):
        """ raw_file: geoLife.hdf5
            list_file: list/training.txt
            trajectory_window: 2048 (假设这是每次读取轨迹点的数量)
        """
        self.raw_file = raw_file 
        self.trajectory_window = trajectory_window
        self.trajectories = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]  
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:
            traj_len = self.h5f[i].shape[0]
            self.trajectories.append(i)  # 添加所有轨迹ID，不再进行长度过滤
            
        # for i in self.trajectories:
        #     if(self.h5f[i].shape[0] > self.trajectory_window):
        #         print(self.h5f[i].shape[0])
           
    def __len__(self):
        """Denotes the total number of trajectories"""
        return len(self.trajectories)

    def __getitem__(self, index):
        traj_id = self.trajectories[index]  # get the trajectory id 
        traj_len = self.h5f[traj_id].shape[0]  # get the number of data points in the trajectory
        
        if traj_len >= self.trajectory_window:
            start_index = np.random.randint(traj_len - self.trajectory_window + 1)  # get the index to read part of the trajectory into memory
            trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
        else:
            # 对于长度小于trajectory_window的轨迹，用最后一个元素进行填充
            last_element = self.h5f[traj_id][-1]  # 获取数据集中的最后一个元素
            padding = np.tile(last_element, (self.trajectory_window - traj_len, 1))  # 创建填充数组
            trajectory_data = np.concatenate((self.h5f[traj_id][:], padding), axis=0)        

        # 确保了数据的维度顺序与模型的输入需求相匹配
        trajectory_data = np.transpose(trajectory_data)  # Assuming 'Latitude', 'Longitude', 'Zero', 'Altitude', 'Days'
        return torch.tensor(trajectory_data, dtype=torch.float)


class RawDatasetSingapore(data.Dataset):
    def __init__(self, raw_file, list_file, trajectory_window):
        """ raw_file: geoLife.hdf5
            list_file: list/training.txt
            trajectory_window: 2048 (假设这是每次读取轨迹点的数量)
        """
        self.raw_file = raw_file 
        self.trajectory_window = trajectory_window
        self.trajectories = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]  
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:
            traj_len = self.h5f[i].shape[0]
            self.trajectories.append(i)  # 添加所有轨迹ID，不再进行长度过滤
            
        # for i in self.trajectories:
        #     if(self.h5f[i].shape[0] > self.trajectory_window):
        #         print(self.h5f[i].shape[0])
           
    def __len__(self):
        """Denotes the total number of trajectories"""
        return len(self.trajectories)

    def __getitem__(self, index):
        traj_id = self.trajectories[index]  # get the trajectory id
        traj_len = self.h5f[traj_id].shape[0]  # get the number of data points in the trajectory
        
        if traj_len >= self.trajectory_window:
            start_index = np.random.randint(traj_len - self.trajectory_window + 1)  # get the index to read part of the trajectory into memory
            trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
        else:
            # 对于长度小于trajectory_window的轨迹，用最后一个元素进行填充
            last_element = self.h5f[traj_id][-1]  # 获取数据集中的最后一个元素
            padding = np.tile(last_element, (self.trajectory_window - traj_len, 1))  # 创建填充数组
            trajectory_data = np.concatenate((self.h5f[traj_id][:], padding), axis=0)        

        # 确保了数据的维度顺序与模型的输入需求相匹配
        trajectory_data = np.transpose(trajectory_data)  # Assuming 'Latitude', 'Longitude', 'Zero', 'Altitude', 'Days'
        return torch.tensor(trajectory_data, dtype=torch.float)
    def close(self):
        """关闭HDF5文件"""
        self.h5f.close()

#  带有标签的新加坡数据集，用于分类任务
class RawDatasetSingaporeForClass(data.Dataset):
    def __init__(self, raw_file, list_file, trajectory_window):
        """
        raw_file: HDF5文件路径，包含所有轨迹数据
        list_file: txt文件，包含轨迹ID列表
        trajectory_window: 每次读取的轨迹点数量
        """
        self.raw_file = raw_file 
        self.trajectory_window = trajectory_window
        self.trajectories = []

        # 读取轨迹ID列表
        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]  # 移除空格和换行符
        
        # 打开HDF5文件
        self.h5f = h5py.File(self.raw_file, 'r')
        
        # 加载所有轨迹ID
        for traj_id in temp:
            traj_len = self.h5f[traj_id].shape[0]
            self.trajectories.append(traj_id)  # 保存轨迹ID

    def __len__(self):
        """返回轨迹总数"""
        return len(self.trajectories)

    def __getitem__(self, index):
        traj_id = self.trajectories[index]  # 获取轨迹ID
        traj_len = self.h5f[traj_id].shape[0]  # 获取轨迹的长度
        
        # 读取轨迹数据并进行填充或截取
        if traj_len >= self.trajectory_window:
            start_index = np.random.randint(traj_len - self.trajectory_window + 1)
            trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
        else:
            # 若轨迹长度小于指定窗口大小，填充数据
            last_element = self.h5f[traj_id][-1]  # 获取最后一个数据点
            padding = np.tile(last_element, (self.trajectory_window - traj_len, 1))
            trajectory_data = np.concatenate((self.h5f[traj_id][:], padding), axis=0)

        # 将数据的维度转置，确保符合模型的输入要求
        trajectory_data = np.transpose(trajectory_data)  # 转置数据
        trajectory_data = torch.tensor(trajectory_data, dtype=torch.float)

        # 提取标签信息（从轨迹ID中提取交通模式）
        label = int(traj_id.split('_')[-1])  # 假设标签是轨迹ID的最后一个部分（0 或 1）

        return trajectory_data, label

    def close(self):
        """关闭HDF5文件"""
        self.h5f.close()



# 雅加达和新加坡共用的翻转数据集操作（包含了原始的和翻转后的)
class RawDatasetSingaporeReverse(data.Dataset):
    def __init__(self, raw_file, list_file, trajectory_window):
        """ raw_file: geoLife.hdf5
            list_file: list/training.txt
            trajectory_window: 2048 (假设这是每次读取轨迹点的数量)
        """
        self.raw_file = raw_file 
        self.trajectory_window = trajectory_window
        self.trajectories = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]  
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp:
            traj_len = self.h5f[i].shape[0]
            self.trajectories.append(i)  # 添加所有轨迹ID，不再进行长度过滤
        
           
    def __len__(self):
        """Denotes the total number of trajectories"""
        return len(self.trajectories)

    def __getitem__(self, index):
        traj_id = self.trajectories[index]  # get the trajectory id
        traj_len = self.h5f[traj_id].shape[0]  # get the number of data points in the trajectory
        
        if traj_len >= self.trajectory_window:
            start_index = np.random.randint(traj_len - self.trajectory_window + 1)  # get the index to read part of the trajectory into memory
            trajectory_data = self.h5f[traj_id][start_index:start_index + self.trajectory_window]
        else:
            # 对于长度小于trajectory_window的轨迹，用最后一个元素进行填充
            last_element = self.h5f[traj_id][-1]  # 获取数据集中的最后一个元素
            padding = np.tile(last_element, (self.trajectory_window - traj_len, 1))  # 创建填充数组
            trajectory_data = np.concatenate((self.h5f[traj_id][:], padding), axis=0)        

        # 确保了数据的维度顺序与模型的输入需求相匹配
        trajectory_data = np.transpose(trajectory_data)  # Assuming 'Latitude', 'Longitude', 'Zero', 'Altitude', 'Days'
        trajectory_reverse_data=trajectory_data[::-1].copy()
        return torch.tensor(trajectory_data, dtype=torch.float),torch.tensor(trajectory_reverse_data, dtype=torch.float)
    def close(self):
        """关闭HDF5文件"""
        self.h5f.close()


class RawXXreverseDataset(data.Dataset):
    ''' RawDataset but returns sequence twice: x, x_reverse '''
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
    
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        original = self.h5f[utt_id][index:index+self.audio_window]
        return original, original[::-1].copy() # reverse

class RawDataset(data.Dataset):
    def __init__(self, raw_file, list_file, audio_window):
        """ raw_file: train-clean-100.h5
            list_file: list/training.txt
            audio_window: 20480
        """
        self.raw_file  = raw_file 
        self.audio_window = audio_window 
        self.utts = []

        with open(list_file) as f:
            temp = f.readlines()
        temp = [x.strip() for x in temp]
        
        self.h5f = h5py.File(self.raw_file, 'r')
        for i in temp: # sanity check
            utt_len = self.h5f[i].shape[0]
            if utt_len > 20480:
                self.utts.append(i)
        """
        with open(index_file) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        self.spk2idx = {}
        for i in content:
            spk = i.split(' ')[0]
            idx = i.split(' ')[1]
            self.spk2idx[spk] = int(idx)
        """
    def __len__(self):
        """Denotes the total number of utterances
        """
        return len(self.utts)

    def __getitem__(self, index):
        utt_id = self.utts[index] # get the utterance id 
        utt_len = self.h5f[utt_id].shape[0] # get the number of data points in the utterance
        index = np.random.randint(utt_len - self.audio_window + 1) # get the index to read part of the utterance into memory 
        #speaker = utt_id.split('-')[0]
        #label   = self.spk2idx[speaker]

        return self.h5f[utt_id][index:index+self.audio_window]

if __name__ == '__main__':
    # 使用示例
    dataset = RawDatasetGeoLife('/home/ubuntu/Data/gch/CPCForGeoLife/geoLife.hdf5', '/home/ubuntu/Data/gch/CPCForGeoLife/geoLifeDataGenerate/geoValList.txt', 2048)
    # train_loader = data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True) # set shuffle to True
    # for batch_idx, data in enumerate(train_loader):
    #     print(data.shape)
        
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[0].shape)
""" 
转置操作的作用：
    这种操作将数据的结构从按行排列（每行是一个时间步）转换为按列排列（每列是一个时间步），
    这种转置在很多时候有助于模型的处理和训练，特别是对于涉及时间序列分析或者空间数据分析的任务。
tensor([[3.9775e+01, 3.9775e+01, 3.9774e+01,  ..., 3.9944e+01, 3.9945e+01,
         3.9946e+01],
        [1.1629e+02, 1.1629e+02, 1.1629e+02,  ..., 1.1680e+02, 1.1680e+02,
         1.1679e+02],
        [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 0.0000e+00,
         0.0000e+00],
        [1.2500e+02, 1.2600e+02, 1.2400e+02,  ..., 1.7000e+02, 1.6400e+02,
         1.5400e+02],
        [3.9917e+04, 3.9917e+04, 3.9917e+04,  ..., 3.9918e+04, 3.9918e+04,
         3.9918e+04]])
torch.Size([5, 2048])
"""
