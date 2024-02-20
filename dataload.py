train_dir = "E:\\数模\\NJU_CPOL_update2308\\NJU_CPOL_update2308\\dBZ"
test_dir = "../data/hotdog/test"

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import os,sys
import h5py
import torch
from torch import nn

class MyDataset(Dataset):
    def __init__(self,path,path2,path3):
        self.path=path
        self.path2=path2
        self.path3=path3
        data_path=[]
        data_path2=[]
        data_path3=[]
        t_data=[]
        t_data2 = []
        t_data3 = []
        tes_data=[]
        for file in os.listdir(self.path):
            dir_path = os.path.join(self.path, file)
            data_path.append(dir_path)
            dir_path = os.path.join(self.path2, file)
            data_path2.append(dir_path)
            dir_path = os.path.join(self.path3, file)
            data_path3.append(dir_path)
        for f in range(len(data_path)):
            fram_path=os.listdir(data_path[f])
            for k in range(len(os.listdir(data_path[f]))-20+1):
                for i in range(10):
                    ddir_path=os.path.join(data_path[f],fram_path[i+k])
                    t_data.append(ddir_path)
                    ddir_path = os.path.join(data_path2[f], fram_path[i + k])
                    t_data2.append(ddir_path)
                    ddir_path = os.path.join(data_path3[f], fram_path[i + k])
                    t_data3.append(ddir_path)
                    tddir_path = os.path.join(data_path[f],fram_path[i+k+10])
                    tes_data.append(tddir_path)
        self.train_path=t_data
        self.train_path2 = t_data2
        self.train_path3 = t_data3
        self.test_path=tes_data



    def __getitem__(self, item):
        train=np.empty([256,256,10],dtype=float)
        train1 = np.empty([256, 256, 10], dtype=float)
        train2 = np.empty([256, 256, 10], dtype=float)
        test=np.empty([256,256,10],dtype=float)
        for i in range(10):
            train[:,:,i]=np.load(self.train_path[item*10+i])/65
            train1[:, :, i]=(np.load(self.train_path2[item*10+i])+1)/7
            train2[:, :, i] = (np.load(self.train_path3[item * 10 + i])+1)/6
            test[:, :, i] = np.load(self.test_path[item * 10 + i])/65

        return train,train1,train2,test

    def __len__(self):
        # print(len(self.train_path)/10)
        return int(len(self.train_path)/10) #返回数据的总个数



# 测试：
# for batch_idx, (inputs, targets) in enumerate(dataset_loader):
#     print(inputs.shape)
#     print(torch.max(inputs[:,:,0:9]))
#     print(torch.max(inputs[:, :, 10:19]))
#     print(torch.max(inputs[:, :, 20:29]))
#     print(torch.sum(inputs))
#     print(torch.max(targets))
#     print(torch.sum(targets))
#     print(targets.shape)

class block_down(nn.Module):

    def __init__(self, inp_channel, out_channel):
        super(block_down, self).__init__()
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class block_up(nn.Module):

    def __init__(self, inp_channel, out_channel, y):
        super(block_up, self).__init__()
        self.up = nn.ConvTranspose2d(inp_channel, out_channel, 2, stride=2)
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)
        self.y = y

    def forward(self, x):
        x = self.up(x)
        x = torch.cat([x, self.y], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class U_net(nn.Module):

    def __init__(self, out_channel):
        super(U_net, self).__init__()
        self.out = nn.Conv2d(64, out_channel, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x,xx,xxx):
        block1 = block_down(10, 64)
        x1_use = block1(x)
        x1 = self.maxpool(x1_use)
        block2 = block_down(64, 128)
        x2_use = block2(x1)
        x2 = self.maxpool(x2_use)
        block3 = block_down(128, 256)
        x3_use = block3(x2)
        x3 = self.maxpool(x3_use)
        block4 = block_down(256, 512)
        x4_use = block4(x3)
        x4 = self.maxpool(x4_use)

        block1 = block_down(10, 64)
        xx1_use = block1(xx)
        xx1 = self.maxpool(xx1_use)
        block2 = block_down(64, 128)
        xx2_use = block2(xx1)
        xx2 = self.maxpool(xx2_use)
        block3 = block_down(128, 256)
        xx3_use = block3(xx2)
        xx3 = self.maxpool(xx3_use)
        block4 = block_down(256, 512)
        xx4_use = block4(xx3)
        xx4 = self.maxpool(xx4_use)

        block1 = block_down(10, 64)
        xxx1_use = block1(xxx)
        xxx1 = self.maxpool(xxx1_use)
        block2 = block_down(64, 128)
        xxx2_use = block2(xxx1)
        xxx2 = self.maxpool(xxx2_use)
        block3 = block_down(128, 256)
        xxx3_use = block3(xxx2)
        xxx3 = self.maxpool(xxx3_use)
        block4 = block_down(256, 512)
        xxx4_use = block4(xxx3)
        xxx4 = self.maxpool(xxx4_use)
        # block5 = block_down(512, 1024)
        # x5 = block5(x4)
        #
        # block6 = block_up(1024, 512, x4_use)
        # x6 = block6(x5)

        x6 = torch.cat((x4, xx4,xxx4), dim=3)  # 模型层拼合！！！当然你的模型中可能不需要~

        block7 = block_up(512, 256, x3_use)
        x7 = block7(x6)
        block8 = block_up(256, 128, x2_use)
        x8 = block8(x7)
        block9 = block_up(128, 64, x1_use)
        x9 = block9(x8)
        x10 = self.out(x9)
        return x10

if __name__ == "__main__":
    dataset = MyDataset('E:\\数模\\NJU_CPOL_update2308\\NJU_CPOL_update2308\\dBZ\\1.0km',
                            r'E:\数模\NJU_CPOL_update2308\NJU_CPOL_update2308\KDP\1.0km',
                            r'E:\数模\NJU_CPOL_update2308\NJU_CPOL_update2308\ZDR\1.0km')

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [20000, 10000])

    if torch.cuda.is_available():
        device = torch.device("cuda")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    train_loader = DataLoader(train_dataset, batch_size=300, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)
    net = U_net(10)
    net=net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    loss_func=loss_func.to(device)

    for t in range(10):
        print("-------第{}轮训练开始-------".format(t+1))
        for inputs1,inputs2,inputs3,targets in train_loader:
            inputs1=inputs1.to(device)
            inputs2 = inputs2.to(device)
            inputs3 = inputs3.to(device)

            prediction1 = net(inputs1,inputs2,inputs3)
            loss1 = loss_func(prediction1, targets)


            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            if t % 100 == 0:
                print('------------训练Loss1 = %.4f' % loss1.data )
        total_test_loss=0

        for inputs1,inputs2,inputs3,targets in test_loader:
            prediction = net(inputs1, inputs2, inputs3)
            loss = loss_func(prediction, targets)
            total_test_loss = total_test_loss + loss.item()
        print("整体测试集上的Loss:{}".format(total_test_loss))





