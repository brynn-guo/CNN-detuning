# lzm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from qutip import (about, basis, destroy, mesolve, ptrace, qeye,
                   tensor, wigner, anim_wigner)
# set a parameter to see animations in line
from matplotlib import rc
import qutip as qp
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 指定GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


#生成dress数据
def dress(wa, wb, J, tlist):
    
	sz1 = tensor(qp.sigmaz(), qp.identity(2))  # 第一个量子比特的σz
	sz2 = tensor(qp.identity(2), qp.sigmaz())  # 第二个量子比特的σz
	sy1 = tensor(qp.sigmay(), qp.identity(2))  # 第一个量子比特的σy
	sy2 = tensor(qp.identity(2), qp.sigmay())  # 第二个量子比特的σy

	# 构建哈密顿量
	H0 = 0.5*wa*sz1 + 0.5*wb*sz2  # 自由哈密顿量

	H_int = J * tensor(qp.sigmay(), qp.sigmay())                
	H = H0 + H_int 


	# 定义初始态为 |00⟩
	psi0 = tensor(basis(2,1), basis(2,0))
	
	# 进行时间演化（无耗散）
	result = mesolve(H, psi0, tlist, [], [])

	# 定义投影算符用于计算布居数
	proj_00 = tensor(basis(2,0).proj(), basis(2,0).proj())
	proj_01 = tensor(basis(2,0).proj(), basis(2,1).proj())
	proj_10 = tensor(basis(2,1).proj(), basis(2,0).proj())
	proj_11 = tensor(basis(2,1).proj(), basis(2,1).proj())

	pop_00 = qp.expect(proj_00, result.states)
	pop_01 = qp.expect(proj_01, result.states)
	pop_10 = qp.expect(proj_10, result.states)
	pop_11 = qp.expect(proj_11, result.states)

	return (pop_00, pop_01, pop_10, pop_11 )


# tlist 从0-10, fre 从1-10
def generate_detuning(width, height, wb, J):
    
	tlist = np.linspace(0,10,height)
	fre_list = np.linspace(0,10,width)

	all = []
	for fre in fre_list:
		wa = fre * 2 * np.pi
		all.append(dress(wa, wb, J, tlist=tlist)[1])

	all = np.array(all)
	return all



# 生成测试数据集
df = pd.read_pickle('data.pkl')

from torch.utils.data import Dataset



class detuningDataset(Dataset):
	def __init__(self, df):
		self.image = df['datas'].values  # 第三列作为输入
		self.params = df.iloc[:, 0:2].values  # 第一和第二列作为输出

	def __len__(self):
		return len(self.image)

	def __getitem__(self, idx):
		image = self.image[idx]
		params = self.params[idx]
		image_tensor = torch.from_numpy(image).float().permute(0, 1)  # (C, H, W)
		image_tensor = image_tensor.unsqueeze(0) 
		params = torch.tensor(params, dtype=torch.float32)
		return  image_tensor, params



# 示例数据集
dataset = detuningDataset(df)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=128,  # 增大批量至显存允许的最大值（测试调整）
    shuffle=True,
    num_workers=8,    # 设置为CPU核心数（如8）
    pin_memory=True,  # 加速数据到GPU的传输
    persistent_workers=True  # 保持worker进程
)

#构建网络
import torch.nn as nn

class ParamPredictor(nn.Module):
    def __init__(self):
        super(ParamPredictor, self).__init__()
        
        # 卷积层处理图像和坐标
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),  # 输入通道数为1，适用于单一特征图
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层预测参数
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 30 * 30, 256),  # 根据实际池化后的尺寸调整输入特征数
            nn.ReLU(),
            nn.Dropout(p=0.5),               # 添加丢弃率防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)                # 输出两个参数，修正注释错误
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(-1, 32 * 30 * 30)     # Flatten tensor to 1D
        out = self.fc_layers(out)
        return out

# 初始化模型
model = ParamPredictor()
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # 添加并行封装
model = model.to(device)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    for batch_idx, (images, params) in enumerate(dataloader):

        images = images.to(device)
        params = params.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, params)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印训练信息
        if batch_idx % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

torch.save(model.cpu().state_dict(), "detuning-v1.pth")