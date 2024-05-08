import pickle
import time
from scipy.io import savemat
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch import nn
from torch.nn import Sequential
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
# 训练集输入矩阵转换为[样本个数，特征维度]格式，目标矩阵转换为[样本个数，目标维度]格式，每个样本变换为1*24*6=144的特征
nwp_train_input=nwp_norm[7*day_point:train_point,:]
power_train_input_LastDay=power[6*day_point:train_point-1*day_point,:]
power_train_input_LastWeek=power[0:train_point-7*day_point,:]
train_feature=np.concatenate([nwp_train_input,power_train_input_LastDay,power_train_input_LastWeek],axis=1)
for i in range(train_point-8*day_point+1):
    feature=np.expand_dims(train_feature[i:i+day_point,:],axis=0)
    if i==0:
        train_input=feature
    else:
        train_input=np.concatenate([train_input,feature],axis=0)
train_input=train_input.reshape([train_point-8*day_point+1,len_lstm*6])
train_target=power[8*day_point-1:train_point,:]
# 训练集输入矩阵转换为[样本个数，特征维度]格式，目标矩阵转换为[样本个数，目标维度]格式，每个样本变换为1*24*6=144的特征
nwp_test_input=nwp_norm[train_point-1*day_point+1:train_point+test_point,:]
power_test_input_LastDay=power[train_point-2*day_point+1:train_point+test_point-1*day_point,:]
power_test_input_LastWeek=power[train_point-8*day_point+1:train_point+test_point-7*day_point,:]
test_feature=np.concatenate([nwp_test_input,power_test_input_LastDay,power_test_input_LastWeek],axis=1)
for i in range(test_point):
    feature=np.expand_dims(test_feature[i:i+day_point,:],axis=0)
    if i==0:
        test_input=feature
    else:
        test_input=np.concatenate([test_input,feature],axis=0)
test_input=test_input.reshape([test_point,len_lstm*6])
test_target=power[train_point:train_point+test_point,:]

# 更改数据格式为tensor,用于网络训练
Train_input = torch.tensor(train_input, dtype=torch.float32)
Train_target = torch.tensor(train_target, dtype=torch.float32)
Test_input = torch.tensor(test_input, dtype=torch.float32)
Test_target = torch.tensor(test_target, dtype=torch.float32)

## 创建网络模型，确定损失函数，优化器，并设置相应超参数
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model1=Sequential(
            nn.Linear(6*24, 200),
            nn.Sigmoid(),
            nn.Linear(200, 32),
            nn.Sigmoid(),
            nn.Linear(32,16),
            nn.Sigmoid(),
            nn.Linear(16,8),
            nn.Tanh(),
            nn.Linear(8,1)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
# 创建预测器对象
model = MLP()
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
end_time_ = time.time()
total_time=end_time_-start_time
## 采用测试集测试模型效果，并计算评价指标
with torch.no_grad():
    Test_output = model(Test_input)
    Test_target = Test_target.to(device0)
    Test_output = Test_output.to(device0)
    Test_target_=Test_target.reshape([365,24])
    Test_output_=Test_output.reshape([365,24])
    rmse_day=((((Test_target_-Test_output_)**2).mean(dim=1))**0.5).mean()
