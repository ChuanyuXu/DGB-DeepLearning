import pickle
import time
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn
# 数据集：power为17250*1功率序列，nwp为17250*4nwp序列
with open('data.pkl','rb') as sjtj:
    [power,nwp]=pickle.load(sjtj)
## 划分训练集与测试集，并将将数据转换为所需格式，设置序列长度为24
# 训练集长度
train_point=8760
# 测试集长度
test_point=8760
# 每个自然日时点数
day_point=24
# lstm序列长度
len_lstm=24
# nwp维度
dim_nwp=np.size(nwp,axis=1)
# nwp归一化，功率已经脱敏，无需归一化
nwp_norm=nwp/np.max(abs(nwp),axis=0)
# 训练集输入矩阵转换为[样本个数，序列长度，特征维度]格式，目标矩阵转换为[样本个数，序列长度，目标维度]格式
nwp_train_input=nwp_norm[7*24:train_point,:].reshape([-1,len_lstm,dim_nwp])
power_train_input_LastDay=power[6*24:train_point-1*24,:].reshape([-1,len_lstm,1])
power_train_input_LastWeek=power[0:train_point-7*24,:].reshape([-1,len_lstm,1])
train_input=np.concatenate([nwp_train_input,power_train_input_LastDay,power_train_input_LastWeek],axis=2)
train_target=power[7*24:train_point,:].reshape([-1,len_lstm,1])
# 测试集输入矩阵转换为[样本个数，序列长度，特征维度]格式，目标矩阵转换为[样本个数，序列长度，目标维度]格式
nwp_test_input=nwp_norm[train_point:train_point+test_point,:].reshape([-1,len_lstm,dim_nwp])
power_test_input_LastDay=power[train_point-1*24:train_point+test_point-1*24,:].reshape([-1,len_lstm,1])
power_test_input_LastWeek=power[train_point-7*24:train_point+test_point-7*24,:].reshape([-1,len_lstm,1])
test_input=np.concatenate([nwp_test_input,power_test_input_LastDay,power_test_input_LastWeek],axis=2)
test_target=power[train_point:train_point+test_point,:].reshape([-1,len_lstm,1])
# 更改数据格式为tensor,用于网络训练
Train_input = torch.tensor(train_input, dtype=torch.float32)
Train_target = torch.tensor(train_target, dtype=torch.float32)
Test_input = torch.tensor(test_input, dtype=torch.float32)
Test_target = torch.tensor(test_target, dtype=torch.float32)

## 创建网络模型，确定损失函数，优化器，并设置相应超参数
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm= nn.LSTM(input_size=6, hidden_size=40, num_layers=3,batch_first=True)
        self.linear = nn.Linear(in_features=40,out_features=1)
    def forward(self, x):
        x,_=self.lstm(x)
        x=self.linear(x)
        return x
# 创建预测器对象
model = LSTM()
# 定义loss的度量方式
loss_fn=nn.MSELoss()
## 其次定义 优化函数,优化函数的学习率为0.0002,betas:用于计算梯度以及梯度平方的运行平均值的系数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
# 定义训练的设备
device=torch.device("cuda")
device0=torch.device("cpu")
# 训练
model = model.to(device)
Train_input=Train_input.to(device)
Train_target=Train_target.to(device)
Test_input=Test_input.to(device)
Test_target=Test_target.to(device)
# 训练的轮数
epoch = 12000
start_time=time.time()
# 添加tensorboard
writer = SummaryWriter("./logs_train")
# 开始训练
for i in range(epoch):
    model.train()
    Train_output=model(Train_input)
    loss=loss_fn(Train_target,Train_output)
    # 梯度归0
    optimizer.zero_grad()
    # 进行反向传播
    loss.backward()
    # 更新网络参数
    optimizer.step()
    ## 打印训练损失
    if (i+1) % 100 == 0:
        end_time = time.time()
        print(end_time - start_time)
        print("第{}轮训练,loss{}".format(i + 1,loss.item()))
        writer.add_scalar("loss_train", loss.item(), i+1)

model.eval()
## 采用测试集测试模型效果，并计算评价指标
with torch.no_grad():
    Test_output = model(Test_input)
    rmse_day=((((Test_target-Test_output)**2).mean(dim=1))**0.5).mean()

