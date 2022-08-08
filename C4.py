import numpy as np
import math
import random
from copy import deepcopy
dict = {...}
import torch
import torch.utils.data  as Data
import torch.nn as nn
import scipy.sparse as sp
from torch.autograd import Variable
from GCN5 import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
def load_files():
    l_d = np.loadtxt("../HOHGCN/lncdata/lnc_dis_association.txt")
    l_m = np.loadtxt("../HOHGCN/lncdata/yuguoxian_lnc_mi.txt")
    lnc_sim=np.loadtxt("../HOHGCN/lncdata/lnc_sim.txt")
    dis_sim=np.loadtxt("../HOHGCN/lncdata/dis_sim_matrix_process.txt")
    m_d=np.loadtxt("../HOHGCN/lncdata/mi_dis.txt")
    micro_sim=np.loadtxt("../HOHGCN/lncdata/micro_sim.txt")
    return l_d,l_m,lnc_sim,dis_sim,m_d,micro_sim
class value_index():
    def __init__(self,num,i,j):
        self.value = num  # save value
        self.value_x = i  # save index of row
        self.value_y = j  # save index of column

def get_num(m1):
    index=[]
    for i in range(len(m1)):
        if m1[i]==1:
            index.append(i)
    return index
def get_sim(m1_dis,m2_dis,dis_sim):
    disease=[]
    for i in range(len(m2_dis)):
        index=dis_sim[m1_dis][m2_dis[i]]
        disease.append(index)
    #print(np.max(disease))
    return np.max(disease)
def get_sim_all(m1_disease,m2_disease,dis_sim):
     disease_sum1=[]
     for i in range(len(m1_disease)):
         temp=get_sim(m1_disease[i],m2_disease,dis_sim)
         disease_sum1.append(temp)
     m1_dis=np.sum(disease_sum1)
     disease_sum2=[]
     for i in range(len(m2_disease)):
         temp=get_sim(m2_disease[i],m1_disease,dis_sim)
         disease_sum2.append(temp)
     m2_dis=np.sum(disease_sum2)
     sum=m1_dis+m2_dis
     return sum

def get_sim_value(m1,m2,dis_sim):
    m1_disease_index=get_num(m1)
    m2_disease_index =get_num(m2)
    m1_disease_len=len(m1_disease_index)
    m2_disease_len=len(m2_disease_index)
    len_sum=m1_disease_len+m2_disease_len
    if m1_disease_len==0 or m2_disease_len==0:
        sim=0
    else :
        m_d_sim=get_sim_all(m1_disease_index,m2_disease_index,dis_sim)
        sim=m_d_sim/(len_sum)
        #print(sim)
    return sim
def get_micro_sim(m_d,dis_sim):
    length=len(m_d)
    n=0
    micro_sim=np.zeros((length,length))
    #print(micro_sim.shape)
    for i in range(length):
       for j in range (length):
           if (i==j):
               micro_sim[i][i] = 1
           else:
            n=n+1
            sim=get_sim_value(m_d[i],m_d[j],dis_sim)
            micro_sim[i][j]=sim
            #print(n)
    return micro_sim
class data_input():
    def __init__(self, value, x, y):
        self.value = value # value 0 or 1
        self.index_x = x  # the row in A_association   A(x,y)
        self.index_y = y  # the column in A_association A(x,y)
    def add_probability_predict(self,probability,predict_value):
        self.probability = probability
        self.predict_value = predict_value

def concatenate(a,b,c,arix1,arix2):
    a_b=np.concatenate((a,b),arix1)
    a_b_c=np.concatenate((a_b,c),arix2)
    return a_b_c
def matrix_concat_feature(save_all_count_A,l_m,lnc_sim,dis_sim,m_d,micro_sim):
      l_m=np.array(l_m)
      lnc_sim=np.array(lnc_sim)
      dis_sim=np.array(dis_sim)
      m_d=np.array(m_d)
      micro_sim=np.array(micro_sim)
      l_d=np.array(save_all_count_A)
      l=concatenate(lnc_sim,l_d,l_m,1,1)
      d=concatenate(l_d.T,dis_sim,m_d.T,1,1)
      m=concatenate(l_m.T,m_d,micro_sim,1,1)
      feature_matrix=concatenate(l,d,m,0,0)
      return feature_matrix


def matrix_concat(l_d, l_m, lnc_sim, dis_sim, m_d,micro_sim):
    l_m=np.array(l_m)
    lnc_sim=np.array(lnc_sim)
    dis_sim=np.array(dis_sim)
    m_d=np.array(m_d)
    micro_sim=np.array(micro_sim)
    l_d=np.array(l_d)
    zero_matrix_1=np.zeros((240,900))
    zero_matrix_2 =zero_matrix_1.T
    zero_matrix_3 = np.zeros((900, 900))
    zero_matrix_4=np.zeros((240,240))
    zero_matrix_5=np.zeros((240,495))
    zero_matrix_6=np.zeros((240,645))
    zero_matrix_7=np.zeros((405,240))
    zero_matrix_8=np.zeros((405,495))
    zero_matrix_9=np.zeros((495,495))
    zero_matrix_10=np.zeros((405,405))

    l_l_matrix=np.concatenate((np.concatenate((lnc_sim,zero_matrix_1),1),np.concatenate((zero_matrix_2,zero_matrix_3),1)),0)
    l_l_matrix=laplacian(l_l_matrix)
    l_l_l_matrix=np.dot(l_l_matrix,l_l_matrix)
    np.savetxt("../HOHGCN/lncdata/l_l_l_path_matrix1.txt", l_l_l_matrix)
    l_d_matrix=np.concatenate((np.concatenate((np.concatenate((zero_matrix_4,l_d),1),zero_matrix_5),1),np.concatenate((zero_matrix_2,zero_matrix_3),1)),0)
    l_d_matrix=laplacian(l_d_matrix)
    l_l_d_matrix=np.dot(l_l_matrix,l_d_matrix)
    np.savetxt("../HOHGCN/lncdata/l_l_d_path_matrix1.txt", l_l_d_matrix)
    l_m_matrix=np.concatenate((np.concatenate((zero_matrix_6,l_m),1),np.concatenate((zero_matrix_2,zero_matrix_3),1)),0)
    l_m_matrix = laplacian(l_m_matrix)
    l_l_m_matrix=np.dot(l_l_matrix,l_m_matrix)
    np.savetxt("../HOHGCN/lncdata/l_l_m_path_matrix1.txt",l_l_m_matrix)
    d_l_matrix=l_d_matrix.T
    d_l_matrix=laplacian(d_l_matrix)
    l_zero=concatenate(zero_matrix_4,zero_matrix_7.T,zero_matrix_5,1,1)
    d_d_zero=concatenate(zero_matrix_7,dis_sim,zero_matrix_8,1,1)
    m_zero=concatenate(zero_matrix_5.T,zero_matrix_8.T,zero_matrix_9,1,1)
    d_l_zero=concatenate(l_d.T,zero_matrix_10,zero_matrix_8,1,1)
    d_d_matrix=concatenate(l_zero,d_d_zero,m_zero,0,0)
    d_d_matrix=laplacian(d_d_matrix)
    d_d_l_matrix = np.dot(d_d_matrix,l_d_matrix.T)
    np.savetxt("../HOHGCN/lncdata/d_d_l_path_matrix1.txt", d_d_l_matrix)
    d_d_d_matrix=np.dot(d_d_matrix,d_d_matrix)
    np.savetxt("../HOHGCN/lncdata/d_d_d_path_matrix1.txt", d_d_d_matrix)
    d_m_zero=concatenate(zero_matrix_7,zero_matrix_10,m_d.T,1,1)
    d_m_matrix=concatenate(l_zero,d_m_zero,m_zero,0,0)
    d_m_matrix=laplacian(d_m_matrix)
    d_d_m_matrix=np.dot(d_d_matrix,d_m_matrix)
    np.savetxt("../HOHGCN/lncdata/d_d_m_path_matrix1.txt", d_d_m_matrix)
    m_l_matrix=l_m_matrix.T
    m_l_matrix=laplacian(m_l_matrix)
    m_d_matrix=d_m_matrix.T
    m_d_matrix=laplacian(m_d_matrix)
    d_zero=concatenate(zero_matrix_7,zero_matrix_10,zero_matrix_8,1,1)
    m_m_zero=concatenate(zero_matrix_5.T,zero_matrix_8.T,micro_sim,1,1)
    m_m_matrix=concatenate(l_zero,d_zero, m_m_zero,0,0)
    return l_l_matrix,l_d_matrix,l_m_matrix,d_l_matrix,d_d_matrix,d_m_matrix,l_l_l_matrix,l_l_d_matrix,l_l_m_matrix,d_d_l_matrix,d_d_d_matrix,d_d_m_matrix

def laplacian(adjacency):
     adjacency_norm=np.array(np.zeros(adjacency.shape))
     for i in range (len(adjacency)):
         for j in range(len(adjacency[0])):
               if adjacency[i][j]!=0:
                 a=np.sum(adjacency[i])
                 b=np.sum(adjacency[:,j])
                 adjacency_norm[i][j]=adjacency[i][j]/math.sqrt(a*b)
               if adjacency[i][j]==0:
                   adjacency_norm[i][j]=0
     adjacency_norm=np.array(adjacency_norm)
     return  adjacency_norm



def select(save_k_associated,k):
    l_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_1.txt")
    l_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_2.txt")
    l_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_3.txt")
    l_l_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_8.txt")
    l_l_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_9.txt")
    l_l_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_10.txt")
    l_toal=np.loadtxt("../HOHGCN/lncdata/ceshiGCN_7.txt")
    l=l_l+l_d+l_m+l_toal+l_l_l+l_l_d+l_l_m
    d_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_4.txt")
    d_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_5.txt")
    d_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_6.txt")
    d_d_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_11.txt")
    d_d_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_12.txt")
    d_d_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_13.txt")
    d=d_m+d_d+d_l+l_toal+d_d_l+d_d_d+d_d_m
    te_l_l = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_1.txt")
    te_l_d = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_2.txt")
    te_l_m = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_3.txt")
    te_l_toal = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_7.txt")
    te_l = te_l_l+te_l_d+te_l_m+te_l_toal  # 1140*600
    te_d_l = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_4.txt")
    te_d_d = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_5.txt")
    te_d_m = np.loadtxt("../HOHGCN/lncdata/ceshi_ceshiGCN_6.txt")
    te_d = te_d_d + te_d_l + te_d_m + te_l_toal  # 1140*600
    return l,d,te_l,te_d


class Beta_Score(nn.Module):
    def __init__(self, in_size, hidden_size=150):
        super(Beta_Score, self).__init__()
        self.linear1= nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh())
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self,z):
        z=z.to(torch.float32)
        w = self.linear1(z)
        w=self.linear(w)
        beta =F.softmax(w, dim=1)
        # print(beta[1])
        one = np.ones((1140,7, 1))
        one = torch.from_numpy(one).float().view(1140,7, 1)
        one=one.to(torch.float32)
        return ((beta+one)* z).sum(1)

def load_data1(data,l,d,BATCHSIZE,l_d):
    x=[]
    y=[]
    for j in range(len(data)):
        temp_save=[]
        x_A=int(data[j][0])
        y_A=int(data[j][1])
        l_node=l[x_A]
        d_node=d[y_A+240]#定位d的位置
        temp_save.append(l_node)
        temp_save.append(d_node)
        x.append([temp_save])
        label=l_d[[x_A],[y_A]]
        y.append(label)
    print(np.array(x).shape)
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))

    torch_dataset = Data.TensorDataset(x, y)
    data1_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCHSIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=0,  # subprocesses for loading data
        drop_last=True
    )
    return data1_loader
def load_data4(data,BATCHSIZE,l_d):
    x=[]
    y=[]
    a = []
    b = []
    for j in range(len(data)):
        x_A=int(data[j][0])
        y_A=int(data[j][1])
        a.append(x_A)
        b.append(y_A)
        label=l_d[[x_A],[y_A]]
        y.append(label)
    a = torch.FloatTensor(np.array(a))
    b = torch.FloatTensor(np.array(b))
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(a,b,y)
    data4_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCHSIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=0,  # subprocesses for loading data
        drop_last=True
    )
    return data4_loader


def load_data2(data,A,Rna_matrix,disease_matrix,BATCH_SIZE,lnc_mi,mi_dis,ld):

    x = []
    y = []
    for j in range(len(data)): 
        temp_save = []  # cat features
        x_A = int(data[j][0]) #坐标
        y_A = int(data[j][1])
        #构造特征矩阵
        rna_disease_mi = np.concatenate((Rna_matrix[x_A],A[x_A],lnc_mi[x_A]),axis=0)
        disease_rna_mi = np.concatenate((disease_matrix[y_A],A[:,y_A],mi_dis[:,y_A]),axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)
        x.append([temp_save])   #特征矩阵
        label=ld[[x_A],[y_A]]
        print(label)
        y.append(label) #真实值
    print(",,,,,,,,,")
    print(np.array(x).shape)
    print("..........")
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    #数据封装
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=0,              # subprocesses for loading data
            drop_last= True
        )
    return data2_loader

def load_data5(data,A,Rna_matrix,disease_matrix,BATCH_SIZE,lnc_mi,mi_dis):

    x = []
    y = []
    for j in range(len(data)): #len(data)=四份正样本的索引下标对
        temp_save = []  # cat features
        x_A = data[j].index_x #坐标
        y_A = data[j].index_y
        #构造特征矩阵
        rna_disease_mi = np.concatenate((Rna_matrix[x_A],A[x_A],lnc_mi[x_A]),axis=0)
        disease_rna_mi = np.concatenate((disease_matrix[y_A],A[:,y_A],mi_dis[:,y_A]),axis=0)
        temp_save.append(rna_disease_mi)
        temp_save.append(disease_rna_mi)
        x.append([temp_save])   #特征矩阵
        y.append(data[j].value) #真实值
    print(",,,,,,,,,")
    print(np.array(x).shape)
    print("..........")
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    #数据封装
    torch_dataset = Data.TensorDataset(x, y)
    data5_loader = Data.DataLoader(
            dataset=torch_dataset,      # torch TensorDataset format
            batch_size=BATCH_SIZE,      # mini batch size
            shuffle=False,              # random shuffle for training
            num_workers=1,              # subprocesses for loading data
            drop_last=True
        )
    return data5_loader
def load_data3(data,label,BATCHSIZE,drop = False):
    x=[]
    y=[]
    for i in range(len(data)):
        low_temp=data[i]
        x.append(low_temp)
        y.append(label[i])
    print(",,,,,,,,,")
    print(np.array(x).shape)
    print("..........")
    print(np.array(y).shape)
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCHSIZE,  # mini batch size
        shuffle=False,  # random shuffle for training
        num_workers=1,  # subprocesses for loading data
        drop_last=True
    )
    return data2_loader
def save_model(model, filename):
    state= model.state_dict()       #一个简单的python的字典对象,将每一层与它的对应参数建立映射关系.(如model的每一层的weights及偏置等等)
    torch.save(state, filename)
class the_model(nn.Module):
    def __init__(self,batch_size):
        super(the_model, self).__init__()
        self.cnn=CNN()
        self.linear=nn.Sequential(nn.Linear(18240,1000),nn.ReLU(),nn.Linear(1000,2))
    def forward(self,x):
        fcm=self.cnn(x)
        #print(fcm.detach().numpy().shape)
        fcm=fcm.view(-1, 32*570)
        #print(fcm.detach().numpy().shape)
        x=self.linear(fcm)
        x= F.softmax(x, dim=1)
        return x

class the_auto_full(nn.Module):
    def __init__(self):
        super(the_auto_full, self).__init__()
        self.linear=nn.Sequential(nn.Linear(1000,500),nn.ReLU(),nn.Linear(500,2))
    def forward(self,x):
        output=self.linear(x)
        return output
class the_GCNmodel(nn.Module):
    def __init__(self):
        super(the_GCNmodel, self).__init__()
        self.cnn=GCN_CNN()
        self.linear=nn.Sequential(nn.Linear(9600, 1000),nn.ReLU(),nn.Linear(1000,2))

    def forward(self, x):
       fcm = self.cnn(x)
       # print(fcm.detach().numpy().shape)
       fcm = fcm.view(-1, 16*2*300)
       # print(fcm.detach().numpy().shape)
       x = self.linear(fcm)
       x = F.softmax(x, dim=1)
       return x

class the_GCN_FULL(nn.Module):
    def __init__(self):
        super(the_GCN_FULL, self).__init__()
        # self.attention1=Alpha_score()

        self.cnn=GCN_CNN1()
        self.linear=nn.Sequential(nn.Linear(1000, 500),nn.ReLU(),nn.Linear(500,2))

    def forward(self,x):

        x1=self.cnn(x)
        x2=self.linear(x1)
        x = F.softmax(x2, dim=1)
        return x1,x

class the_GCN_FULL2(nn.Module):
    def __init__(self,inputsize1):
        super(the_GCN_FULL2, self).__init__()
        self.attention1=Beta_Score(inputsize1)
        self.attention2=Beta_Score(inputsize1)
        self.tanh=nn.Tanh()
        self.cnn = GCN_CNN1()
        self.linear = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 2))
    def forward(self, A, B, C, D, E, F1,G,H,I,J,K,L,M,x_, y_):
        # input_size=len(A[0])
        A=torch.from_numpy(A).double()
        B = torch.from_numpy(B).double()
        C = torch.from_numpy(C).double()
        D = torch.from_numpy(D).double()
        E = torch.from_numpy(E).double()
        F1 = torch.from_numpy(F1).double()
        G = torch.from_numpy(G).double()
        H = torch.from_numpy(H).double()
        I = torch.from_numpy(I).double()
        J = torch.from_numpy(J).double()
        K = torch.from_numpy(K).double()
        L = torch.from_numpy(L).double()
        M = torch.from_numpy(M).double()
        l=torch.stack([A,B,C,D,H,I,J], dim=1)

        d=torch.stack([E,F1,G,D,K,L,M],dim=1)

        l=self.attention1(l)

        d=self.attention2(d)

        x_ = torch.Tensor.numpy(x_)
        y_ = torch.Tensor.numpy(y_)
        x_node = l[x_]
        y_node = d[y_ + 240]
        x_node = x_node.unsqueeze(1)
        y_node = y_node.unsqueeze(1)
        x = np.concatenate((x_node.detach().numpy(), y_node.detach().numpy()), 1)
        x = torch.FloatTensor(np.array(x))
        x = x.unsqueeze(1)
        x1 = self.cnn(x)
        x2 = self.linear(x1)
        # x3 = F.softmax(x2, dim=1)
        return x2



class the_gcn_full(nn.Module):
    def __init__(self):
        super(the_gcn_full,self).__init__()
        self.linear = nn.Sequential(nn.Linear(1200,600),nn.ReLU(),nn.Linear(600,2))
    def forward(self,x):
        x=x.reshape(50,1200)
        x1=self.linear(x)
        x1=F.softmax(x1,dim=1)
        return x,x1

def train(EPOCH,batchsize,LR):
    l_d1=np.loadtxt("../HOHGCN/lncdata/lnc_dis_association.txt")
    tr = np.loadtxt("../HOHGCN/lncdata/w15_tr.txt")
    te = np.loadtxt("../HOHGCN/lncdata/w15_te.txt")
    in_size1 = 1140
    in_size2 = 300
    output_size = 1140
    H_number = 4
    l_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_1.txt")
    l_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_2.txt")
    l_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_3.txt")
    l_toal = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_7.txt")
    d_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_4.txt")
    d_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_5.txt")
    d_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_6.txt")
    l_l_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_8.txt")
    l_l_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_9.txt")
    l_l_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_10.txt")
    d_d_l = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_11.txt")
    d_d_d = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_12.txt")
    d_d_m = np.loadtxt("../HOHGCN/lncdata/ceshiGCN_13.txt")
    model1=the_GCN_FULL2(in_size2)
    optimizer = torch.optim.Adam(model1.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    model1.train()
    train_loader = load_data4(tr, batchsize, l_d1)
    for epoch in range(EPOCH):
        for step, (a,b,y) in enumerate(train_loader):
            model1.train()
            x1 = Variable(a) #转化成Variable类型
            X2 = Variable(b)
            b_y = Variable(y)
            output = model1(l_l, l_d, l_m, l_toal, d_l, d_d, d_m,l_l_l,l_l_d,l_l_m,d_d_l,d_d_d,d_d_m,x1,X2)
            loss = loss_func(output, b_y.squeeze()) #损失
            optimizer.zero_grad() #梯度清零
            loss.backward()       #反向传播
            optimizer.step()      #参数更新
        print(step)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
    if (epoch == EPOCH-1):
            save_model(model1, r'D:\zy_python\HOHGCN\lncdata\1-gcn3model_%d.pkl' % epoch)  # 保存模型去掉kk
    model1.eval()
    same_number = 0.0
    same_number_length = 0.0
    a1=[]
    test_loader = load_data4(te, batchsize, l_d1)  # batch是1 换成50试试
    for step, (a,b,y) in enumerate(test_loader):
        x1=Variable(a)
        x2=Variable(b)
        dev_y =Variable(y)
        dev_output =model1(l_l, l_d, l_m, l_toal, d_l, d_d, d_m,l_l_l,l_l_d,l_l_m,d_d_l,d_d_d,d_d_m,x1,x2)
        dev_output=F.softmax(dev_output,dim=1)
        pred_data=np.array(dev_output.detach().numpy()) 
        pred_data1 = []
        pred_data1.append(pred_data[:,1])  #只保存预测为有关联的概率
        pred_y = torch.max(dev_output, 1)[1].data.squeeze().int() 
        a1.append(pred_data1)
        same_number += sum(np.array(pred_y) == np.array(dev_y)) #统计预测正确的样本个数
        same_number_length += float(dev_y.size(0))
    same_number_length = float(same_number_length)
    same_number = float(same_number)
    accuracy = same_number / same_number_length  #计算精确度
    print('Epoch: ', epoch, '| test accuracy: %.2f' % accuracy)
    return a1 #返回得分列表

def train2(model,epoch,lr,test1_loader,train1_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #损失函数
    loss_func = nn.CrossEntropyLoss()
    model.train()
    for i in range(epoch):
        for _, (x, y) in enumerate(train1_loader):
            b_y = Variable(y)
            output = model(b_x)
            loss = loss_func(output, b_y) #损失
            optimizer.zero_grad() #梯度清零
            loss.backward()       #反向传播
            optimizer.step()      #参数更新
        print('Epoch: ', i, '| train loss: %.4f' % loss.item())
        if (i == epoch - 1):
            save_model(model,r'../HOHGCN/lncdata/7_right_full_model_%d.pkl' % epoch)
    model.eval()
    model.load_state_dict(torch.load('../HOHGCN/lncdata/7_full_model_80.pkl'))
    same_number = 0.0
    same_number_length =0.0
    a = []
    pred_data2 = []
    for _, (xx, yy) in enumerate(test1_loader):
        print(xx.dtype)
        dev_x = Variable(xx)
        dev_y = yy.int()
        dev_output = model(dev_x)
        dev_output=F.softmax(dev_output,dim=1)
        pred_data = np.array(dev_output.detach().numpy())  # 把经过模型得出的结果变成数组类型
        pred_data1 = []
        pred_data1.append(pred_data[:, 1])  # 只保存预测为有关联的概率
        pred_y = torch.max(dev_output, 1)[1].data.squeeze().int()  # 获取每行结果的最大值对应的下标
        a.append(pred_data1)  # 把预测为有关联的概率存到列表里。
        same_number += sum(np.array(pred_y) == np.array(dev_y))  # 统计预测正确的样本个数
        same_number_length += float(dev_y.size(0))
    same_number_length = float(same_number_length)
    same_number = float(same_number)
    accuracy = same_number / same_number_length  # 计算精确度
    #np.savetxt("../HOHGCN/lncdata/test1_label",pred_data2)
    print('Epoch: ', epoch,'| test accuracy: %.2f' % accuracy)
    return a  # 返回得分列表
  
    optimizer = torch.optim.Adam(model1.parameters(), lr=LR)
    #损失函数
    loss_func = nn.CrossEntropyLoss()
    #训练
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            model1.train()
            b_x = Variable(x) #转化成Variable类型
            b_y = Variable(y)
            output = model1(b_x)
            loss = loss_func(output, b_y.squeeze()) #损失
            optimizer.zero_grad() #梯度清零
            loss.backward()       #反向传播
            optimizer.step()      #参数更新
        print(step)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
    if (epoch == EPOCH-1):
            save_model(model1, r'D:\zy_python\HOHGCN\lncdata\mixmodel_%d.pkl' % epoch)  # 保存模型
    model1.eval()
    same_number = 0.0
    same_number_length = 0.0
    # #model.load_state_dict(torch.load('../HOHGCN/lncdata/model_49.pkl')) #加载模型
    a=[]
    pred_data1=[]
    for step,(x, y) in enumerate(test_loader):
        x = torch.FloatTensor(np.array(x))
        dev_x=Variable(x)
        dev_y = y
        dev_output = model1(dev_x)
        pred_data=np.array(dev_output.detach().numpy()) 
        pred_data1 = []
        pred_data1.append(pred_data[:,1])  
        pred_y = torch.max(dev_output, 1)[1].data.squeeze().int() 
        a.append(pred_data1)    
        same_number += sum(np.array(pred_y) == np.array(dev_y)) 
        same_number_length += float(dev_y.size(0))
    same_number_length = float(same_number_length)
    same_number = float(same_number)
    accuracy = same_number / same_number_length  
    print('Epoch: ', epoch, '| test accuracy: %.2f' % accuracy)
    return a # 返回得分列表

    #定义优化器
    #optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-8)
    optimizer = torch.optim.Adam(model1.parameters(), lr=LR)
    #损失函数
    loss_func = nn.CrossEntropyLoss()
    #训练
    for epoch in range(EPOCH):
        #attention()
        low_train = np.zeros(shape=(0, 1000))  # 降到1000维*
        low_label = [] #*
        #加上数据封装
        for step, (x, y) in enumerate(train_loader):
            model1.train()
            b_x = Variable(x) #转化成Variable类型
            b_y = Variable(y)
            low,output = model1(b_x)
            loss = loss_func(output, b_y.squeeze()) #损失
            optimizer.zero_grad() #梯度清零
            loss.backward()       #反向传播
            optimizer.step()      #参数更新
            label = y.view(-1, 1)  #*
            low=low.detach().cpu().numpy()#*
            low_train=np.vstack((low_train,low))#*
            low_label.extend(label)#*
        print(step)
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
    if (epoch == EPOCH-1):
            save_model(model1, r'D:\zy_python\HOHGCN\lncdata\2-gcn3model_%d.pkl' % epoch)  
    model1.eval()
    same_number = 0.0
    same_number_length = 0.0
    a=[]
    pred_data1=[]
    low_train = np.zeros(shape=(0, 1000))  # 降到1000维
    low_label = []
    i=0
    for step,(x, y) in enumerate(test_loader):
        x = torch.FloatTensor(np.array(x))
        dev_x=Variable(x)
        dev_y = y
        print(i)
        i=i+1
        tlow,dev_output = model1(dev_x)
        tlow = tlow.detach().cpu().numpy()
        low_train = np.vstack((low_train, tlow))
        dev_yy=y.view(-1,1)
        low_label.extend(dev_yy)#
        pred_data=np.array(dev_output.detach().numpy()) #把经过模型得出的结果变成数组类型
        pred_data1 = []
        pred_data1.append(pred_data[:,1])  #只保存预测为有关联的概率
        pred_y = torch.max(dev_output, 1)[1].data.squeeze().int() #获取每行结果的最大值对应的下标
        #print(torch.typename(pred_y))
        a.append(pred_data1)    #把预测为有关联的概率存到列表里。
        same_number += sum(np.array(pred_y) == np.array(dev_y)) #统计预测正确的样本个数
        same_number_length += float(dev_y.size(0))
    same_number_length = float(same_number_length)
    same_number = float(same_number)
    accuracy = same_number / same_number_length  #计算精确度
    print('Epoch: ', epoch, '| test accuracy: %.2f' % accuracy)
    # np.savetxt("../HOHGCN/lncdata/1_gcnfull_lowtest.txt", low_train)
    # np.savetxt("../HOHGCN/lncdata/1_gcnfull_lowtestlabel.txt", low_label)
    return a #返回得分
def A_label(A,train_data):
    sum=0
    B=A
    for k in range(len(train_data)):
        x1 = int(train_data[k][0])
        y1 = int(train_data[k][1])
        B[x1][y1]=-1    #在原始的疾病和incrna的基础上，把训练集的位置都设置成-1
        sum=sum+1       #统计了一下参与训练的样本个数
    print(sum)
    return B            #返回标签矩阵
def B_label(A,train_data):
    sum=0
    B=A
    for k in range(len(train_data)):
        x1 = train_data[k].index_x
        y1 = train_data[k].index_y
        B[x1][y1]=-1    
        sum=sum+1  
    print(sum)
    return B            
def main_task(k):
    l_d, l_m, lnc_sim, dis_sim, m_d, micro_sim=load_files()
    epoch=120
    EPOCH=3000
    batchsize = 50
    BATCHSIZE=1
    lr=0.0001
    for i in range(1):
        #lnc_sim=get_micro_sim(save_all_count[i],dis_sim)
        tr = np.loadtxt("D:/zy_python/HOHGCN/lncdata/train_data3.txt")
        te=np.loadtxt("D:/zy_python/HOHGCN/lncdata/test_data3.txt")
        tld=np.loadtxt("D:/zy_python/HOHGCN/lncdata/tld.txt")
        lnc_sim = get_micro_sim(l_d, dis_sim)
        l_l_matrix, l_d_matrix, l_m_matrix, d_l_matrix, d_d_matrix \
            , d_m_matrix,l_l_l_matrix,l_l_d_matrix,l_l_m_matrix\
            ,d_d_l_matrix,d_d_d_matrix,d_d_m_matrix= matrix_concat(tld, l_m, lnc_sim, dis_sim, m_d, micro_sim)
        lnc_sim, dis_sim, m_d, micro_sim)
        np.savetxt("../HOHGCN/lncdata/feature.txt",feature_matrix)
        np.savetxt("../HOHGCN/lncdata/ld.txt",l_d_matrix)
        feature_matrix1=matrix_concat_feature(tld, l_m, lnc_sim, dis_sim, m_d, micro_sim)
        feature_matrix1=laplacian(feature_matrix1)
        AA = np.loadtxt("../HOHGCN/lncdata/lnc_dis_association.txt")  
        train1(EPOCH, feature_matrix, l_l_matrix, 1)
        print(",,,,,,,")
        train1(EPOCH, feature_matrix, l_d_matrix, 2)
        print(",,,,,,,")
        train1(EPOCH, feature_matrix, l_m_matrix, 3)
        print(",,,,,,,")
        train1(EPOCH, feature_matrix, d_l_matrix, 4)
        print(",,,,,,,")
        train1(EPOCH, feature_matrix, d_d_matrix, 5)
        print(",,,,,,,")
        train1(EPOCH, feature_matrix, d_m_matrix,6)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix,feature_matrix1, 7)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, l_l_l_matrix, 8)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, l_l_d_matrix, 9)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, l_l_m_matrix, 10)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, d_d_l_matrix, 11)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, d_d_d_matrix, 12)
        print(",,,,,,,,")
        train1(EPOCH, feature_matrix, d_d_m_matrix, 13)
        print(",,,,,,,,")
        b = train(epoch,batchsize,lr)
        test_data = te
        train_data = tr
        print('填回矩阵')
        SCORE = l_d
        print(np.array(b).shape)
        z = np.reshape(b, (1, -1))
        for k in range(len(te)):
            x1 = int(test_data[k][0])
            y1 = int(test_data[k][1])
            c = z[0]
            SCORE[x1][y1] = c[k] 
        for p in range(len(tr)):
            SCORE[int(train_data[p][0])][int(train_data[p][1])] = -1 
        print(SCORE)
        np.savetxt("../HOHGCN/lncdata/25_ceshi_%d.txt" % epoch, SCORE)
        label = A_label(AA, train_data)  
        np.savetxt("../HOHGCN/lncdata/25_ceshi1label_%d.txt" % epoch, label)
    return SCORE,label 
def main_2_task(k):
    l_d, l_m, lnc_sim, dis_sim, m_d, micro_sim = load_files()
    epoch=100
    epoch1=50
    batchsize = 50
    lr = 0.0005
    for i in range(1):
      tr=np.loadtxt("../HOHGCN/lncdata/w15_tr.txt")
      te=np.loadtxt("../HOHGCN/lncdata/w15_te.txt")
      tld=np.loadtxt("../HOHGCN/lncdata/w15_tld.txt")
      lnc_sim = get_micro_sim(tld, dis_sim)
      train_loader = load_data2(tr, tld, lnc_sim, dis_sim, batchsize, l_m, m_d,l_d)
      test_loader = load_data2(te, tld, lnc_sim, dis_sim, batchsize, l_m, m_d,l_d)
      auto(train_loader, epoch, test_loader)
      low_feature = np.loadtxt("../HOHGCN/lncdata/train_low_feature_w1.txt")
      low_label = np.loadtxt("../HOHGCN/lncdata/train_low_label_w1.txt")
      train1_loader = load_data3(low_feature, low_label, batchsize)
      tlow_feature = np.loadtxt("../HOHGCN/lncdata/test_encode_w8.txt")
      tlow_label = np.loadtxt("../HOHGCN/lncdata/test_label_w8.txt")
      test1_loader = load_data3(tlow_feature, tlow_label, batchsize)
      model = the_auto_full()
      model.train()
      a = train2(model,epoch1,lr, test1_loader,train1_loader)
      test_data = te
      train_data = tr
      AA = np.loadtxt("../HOHGCN/lncdata/lnc_dis_association.txt")
      C_SCORE = l_d
      z = np.reshape(a, (1, -1)) 
      c = []
      for k in range(len(te)):
          x1 = int(test_data[k][0])
          y1 = int(test_data[k][1])
          c = z[0]
          C_SCORE[x1][y1] = c[k] 
      for p in range(len(tr)):
          C_SCORE[int(train_data[p][0])][int(train_data[p][1])] = -1  
      np.savetxt("../HOHGCN/lncdata/test_score_ce1225.txt", C_SCORE)
      labelmarix = A_label(AA, train_data)  
      np.savetxt("../HOHGCN/lncdata/test_labelmatrix_ce1225.txt", labelmarix)
    return C_SCORE, labelmarix  

if __name__ == '__main__':
     SCORE, labelmarix = main_task(5)
     f = Count_valid_data(SCORE)
     TPR, FPR, P = caculate_TPR_FPR(SCORE,f,labelmarix)
     curve(FPR,TPR,P)
