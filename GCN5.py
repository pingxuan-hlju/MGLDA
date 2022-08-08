from numpy import *
from torch.nn.parameter import Parameter
import torch.nn.functional as F
dict = {...}
import torch
import torch.utils.data  as Data
import torch.nn as nn
import scipy.sparse as sp
from torch.autograd import Variable
in_features=1140
in_hiden=600
in_hiden1=900
out_features=300
out1_features=600
epoch=300
epoch1=100
lr=0.0001
L_L_matrix_laplacian=np.loadtxt("D:\zy_python\HOHGCN\lncdata\l_d_path_matrix.txt")
l_d_matrix_laplacian=np.loadtxt("../HOHGCN/lncdata/l_d_path_matrix.txt")
d_l_matrix_laplacian = np.loadtxt("../HOHGCN/lncdata/d_l_path_matrix.txt")
m_m_matrix_laplacian = np.loadtxt("../HOHGCN/lncdata/m_m_path_matrix.txt")
l_m_matrix_laplacian = np.loadtxt("../HOHGCN/lncdata/l_m_path_matrix.txt")
d_d_matrix_laplacian = np.loadtxt("../HOHGCN/lncdata/d_d_path_matrix.txt")
d_m_matrix_laplacian = np.loadtxt("../HOHGCN/lncdata/d_m_path_matrix.txt")
feature_matrix=np.loadtxt("../HOHGCN/lncdata/feature_matrix.txt")
class GraphConvolution(nn.Module):
    def __init__(self, in_size, out_size,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.FloatTensor(in_size, out_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_size))  # tensor类型转换成parameter类型的偏置，变成可以学习的参数
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#均匀分布初始化
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X, A):

        support=torch.spmm(A,X) #A*X
        result = torch.mm(support, self.weight)  # A*X*W
        if self.bias is not None:
            return result + self.bias
        else:
            return result
#两个图卷积类构成的双层图卷积
class GCN(nn.Module):
    def __init__(self, in_features,in_hiden, out_features):    
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, in_hiden) #1140*600
        self.gc2 = GraphConvolution(in_hiden,out_features) #1140*300
        self.gc3 = nn.Conv1d(in_channels=900,out_channels=300,kernel_size=1,stride=1)
        self.gc4 = GraphConvolution(out_features,in_hiden)
        self.gc5 = GraphConvolution(in_hiden,in_features)
    def forward(self, x, A):
        x1 =self.gc1(x, A)
        x1=torch.relu(x1)
        x2 =self.gc2(x1, A)
        x=torch.cat([x2,x1],1)
        x=x.reshape(1,1140,900)
        x=x.permute(0,2,1)
        x=self.gc3(x)
        x=x.reshape(300,1140)
        x=x.permute(1,0)
        x1=self.gc4(x,A)
        x1=torch.sigmoid(self.gc5(x1,A))
        return x,x1

def train1(epoch,feature_matrix,A,name):

 model = GCN(in_features=in_features,
                in_hiden=in_hiden,
                out_features=out_features
                )
 model.train()
 feature_matrix = torch.from_numpy(feature_matrix).float().view(in_features, in_features)
 A = torch.from_numpy(A).float().view(in_features, in_features)
 GCN_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 loss_function_E = nn.MSELoss()
 for epoch in range(epoch):

    Z,x_hat= model(feature_matrix,A)                 # 通过G得到（A，X）的编码结果Z
    # x_hat = decoder(Z,A)          # 解码
    GCN_loss = loss_function_E(x_hat,feature_matrix)       # 计算重构误差
    GCN_optimizer.zero_grad()
    GCN_loss.backward()
    GCN_optimizer.step()
    print('Epoch: ', epoch, '| train G_loss: %.10f' % GCN_loss.item())
    Z = Z.data.cpu()
 result = Z.data.cpu().numpy()
 np.savetxt("../HOHGCN/lncdata/ceshiGCN_"+str(name)+".txt",result)
 model.eval()
 Z ,X_hat= model(feature_matrix, A)
 result1 = Z.data.cpu().numpy()
 np.savetxt("../HOHGCN/lncdata/ceshi_ceshiGCN_" + str(name) + ".txt", result1)

# #节点属性的卷积层面。
class CNN(nn.Module): 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=2,  # filter size
                stride=2,  # filter movement/step
                padding=1, # padding
            ),                                        
            nn.BatchNorm2d(16),
            nn.ReLU(True),  # activation
            nn.MaxPool2d(kernel_size=(1,2), stride=1),
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2,stride=2,padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(2,stride=2),  #50,32,1,143
        )
    def forward(self,x):
        x = self.conv1(x)
        return x
class GCN_CNN(nn.Module):
    def __init__(self):
        super(GCN_CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,  kernel_size=2,  stride=2,  padding=1,),#(2*301)
            nn.BatchNorm2d(16),
            nn.ReLU(True),  # activation
            nn.MaxPool2d(kernel_size=(1,2),stride=1),   #2*300
        )
    def forward(self,x):
        x = self.conv1(x)
        return x
class GCN_CNN1(nn.Module):
    def __init__(self):
        super(GCN_CNN1,self).__init__()
        self.f1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=2,stride=2,padding=1),#2*301
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2),stride=1) #2*300
        )
        self.f2=nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=2,stride=2,padding=1,),#2*151
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=1),  #50,32,1,150
        )
        self.f3=nn.Sequential(nn.Linear(32*1*75,1000))
    def forward(self, x):
        x=self.f1(x)
        x=self.f2(x)
        x=x.view(x.size(0),-1)
        x=self.f3(x)
        return x

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=2, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1,2), stride=1)  
        )
        self.encoder1=nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*1*1*285,2000)
        )
        self.de_fc = nn.Linear(2000, 32*1*1*285)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(True))
        self.decoder1=nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh())
    def forward(self, x):
        encode = self.encoder(x)  
        encode1=self.encoder1(encode)   
        encode_flat = encode1.view(encode1.size(0),-1)   
        encode2 = self.fc(encode_flat)        

        de_encode = self.de_fc(encode2)   
        de_encode = de_encode+encode_flat
        de_encode = de_encode.reshape(50,32,1,285)  
        de_encode = de_encode + encode1
        decode = self.decoder(de_encode)
        decode=decode+encode
        decode=self.decoder1(decode)
        return encode2, decode

def save_model(model, filename):
    state= model.state_dict()
    torch.save(state, filename)


def auto(train_loader, EPOCH, test_loader):
    model = autoencoder()
    auto_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    auto_function = nn.MSELoss()
    for epoch in range(EPOCH):
        auto_loss = 0
        low_train = np.zeros(shape=(0, 2000))
        low_label = []
        for (im, label) in train_loader:
            model.train()
            label = label.view(-1, 1)
            im = Variable(im)
            encode, decode = model(im)
            loss = auto_function(decode, im)
            auto_optimizer.zero_grad()
            loss.backward()
            auto_optimizer.step()
            auto_loss += loss.item()
            print
            encode = encode.detach().cpu().numpy()
            low_train = np.vstack((low_train, encode))
            low_label.extend(label)
        print(auto_loss)
        print(np.array(low_label).shape)
        print(np.array(low_train).shape)
        if (epoch == EPOCH - 1):
            save_model(model, r'../HOHGCN/lncdata/7_right_encode_model_%d.pkl' % EPOCH)
    np.savetxt("../HOHGCN/lncdata/train_low_feature_w7.txt", low_train)
    np.savetxt("../HOHGCN/lncdata/train_low_label_w7.txt", low_label)
    model.load_state_dict(torch.load('../HOHGCN/lncdata/7_encode_model_80.pkl'))
    model.eval()
    low_train = np.zeros(shape=(0, 1000))
    # low_train1=[]
    low_label = []
    for (xx, yy) in test_loader:
        dev_xx = Variable(xx)
        dev_yy = yy.view(-1, 1)
        #print(dev_yy)
        encode, decode = model(dev_xx)
        encode = encode.detach().cpu().numpy()
        low_train = np.vstack((low_train, encode))
        low_label.extend(dev_yy)
    print(np.array(low_label).shape)
    print(np.array(low_train).shape)
    np.savetxt("../HOHGCN/lncdata/test_encode_w8.txt", low_train)
    np.savetxt("../HOHGCN/lncdata/test_label_w8.txt", low_label)