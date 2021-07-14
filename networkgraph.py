from models import SA
from models import Multi_re
from net_utils import MLP
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
import signal

#def l2norm(X, dim, eps=1e-8):
    #"""L2-normalize columns of X
    #"""
    #norm = torch.pow(X, 2).sum(dim=dim).sqrt() + eps
    # X = torch.div(X, norm)
    #X = X / norm.unsqueeze(3)
    #return X
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class Self_att(nn.Module):    #自注意得到每个区域、单词的权重
    def __init__(self, arg):
        super(Self_att, self).__init__()

        self.f1 = nn.Linear(arg.GRU_hidden, int(arg.GRU_hidden / 2))
        self.f2 = nn.Linear(int(arg.GRU_hidden / 2), 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):  #x:n_x,n,d  mask:n_x,1,1,n
        mask = mask.squeeze(1).squeeze(1)        #n_x,n
        att = self.f1(x)
        att = self.tanh(att)
        att = self.dropout(att)
        att = self.f2(att)     #n_x,n,1

        b = x.size(0)
        n = x.size(1)
        att = att.view(b, n).masked_fill(mask, -1e9)
        att = F.softmax(att, dim=1).view(b, n, 1)    #n_x,n,1
        return att


class Rs_GCNI(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCNI, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=(1,1), stride=(1,1), padding=(0,0))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=(1,1), stride=(1,1), padding=(0,0)),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=(1,1), stride=(1,1), padding=(0,0))
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)
        self.linearg = nn.Linear(self.in_channels, self.inter_channels)
        self.linearW = nn.Linear(self.inter_channels,self.in_channels )
        #self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         #kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        #self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             #kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, v,imgD,txtD):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size0 = v.size(0)
        batch_size1 = v.size(1)
        v = v.permute(0,3,2,1)  #n_x,d,n,n_y


        g_v = self.g(v).view(batch_size0, self.inter_channels, -1,batch_size1)
        g_v = g_v.permute(0,3,2,1)   #n_x,n_y,n,d
        #theta_v = self.theta(v).view(batch_size0, self.inter_channels, -1,batch_size1)
        #theta_v = theta_v.permute(0,3,2,1)  # n_x,n_y,n,d
        #phi_v = self.phi(v).view(batch_size0, self.inter_channels, -1,batch_size1)  # n_x,d,n
        #phi_v = phi_v.permute(0, 3, 1, 2)  # n_x,n_y,d,n

        #R = torch.matmul(l2norm(theta_v, 3), l2norm(phi_v, 3).permute(0, 1, 3, 2))  # n_x,n_y,n,n

        R=torch.matmul(l2norm(imgD,3),l2norm(imgD,3).permute(0,1,3,2))  #n_x,n_y,n,n
        N = R.size(-1)
        R_div_C = R / N
        #print(R_div_C.shape)  #返回张量的size
        g_f=R[3,16,25,9]







        y = torch.matmul(R_div_C, g_v)    #n_x,n_y,n,d
        y = y.permute(0, 3,2,1).contiguous()     #n_x,d,n,n_y
        y = y.view(batch_size0,self.inter_channels, -1,batch_size1)
        W_y = self.W(y)   #n_x,d,n,n_y
        v_star = W_y + v
        v_star = v_star.permute(0,3,2,1)  #n_x,n_y,n,d

        return v_star,g_f
class Rs_GCNU(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCNU, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=(1,1), stride=(1,1), padding=(0,0))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=(1,1), stride=(1,1), padding=(0,0)),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=(1,1), stride=(1,1), padding=(0,0))
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)
        self.theta = None
        self.phi = None
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=(1,1), stride=(1,1), padding=(0,0))




    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size0 = v.size(0)
        batch_size1 = v.size(1)
        v = v.permute(0,3,2,1)  #n_x,d,n,n_y


        g_v = self.g(v).view(batch_size0, self.inter_channels, -1,batch_size1)
        g_v = g_v.permute(0,3,2,1)   #n_x,n_y,n,d
        theta_v = self.theta(v).view(batch_size0, self.inter_channels, -1,batch_size1)
        theta_v = theta_v.permute(0, 3,2,1)  # n_x,n_y,n,d
        phi_v = self.phi(v).view(batch_size0, self.inter_channels, -1,batch_size1)  # n_x,d,n,n_y
        phi_v=phi_v.permute(0,3,1,2)



        R = torch.matmul(theta_v, phi_v)  #n_x,n_y,n,n
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)    #n_x,n_y,n,d
        y = y.permute(0, 3,2,1).contiguous()     #n_x,d,n,n_y
        y = y.view(batch_size0,self.inter_channels, -1,batch_size1)
        W_y = self.W(y)   #n_x,d,n,n_y
        v_star = W_y + v
        v_star = v_star.permute(0,3,2,1)  #n_x,n_y,n,d

        return v_star
class Rs_GCN(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCN, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None


        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)




    def forward(self, v):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size = v.size(0)
        v = v.permute(0, 2, 1)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)   #n_x,d,n
        g_v = g_v.permute(0, 2, 1)         #n_x,n,d

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)   #n_x,n,d
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)    #n_x,d,n
        R = torch.matmul(theta_v, phi_v)   #n_x,n,n
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)    #n_x,n,d
        y = y.permute(0, 2, 1).contiguous()     #n_x,d,n
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)   #n_x,d,n
        v_star = W_y + v   #n_x,d,n
        v_star = v_star.permute(0, 2, 1)

        return v_star




class Rs_GCNS(nn.Module):

    def __init__(self, in_channels, inter_channels, bn_layer=True):
        super(Rs_GCNS, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=(1,1), stride=(1,1), padding=(0,0))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=(1,1), stride=(1,1), padding=(0,0)),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=(1,1), stride=(1,1), padding=(0,0))
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

    def forward(self,  v,  imgD, txtD):
        '''
        :param v: (B, D, N)
        :return:
        '''
        batch_size0 = v.size(0)
        batch_size1 = v.size(1)
        v = v.permute(0, 3,2,1)

        g_v = self.g(v).view(batch_size0, self.inter_channels, -1,batch_size1 )
        g_v = g_v.permute(0, 3,2,1)   #n_x,n_y,m,d

        R = torch.matmul(l2norm(txtD,3), l2norm(txtD,3).permute(0, 1, 3, 2))  # n_x,n_y,m,m
        N = R.size(-1)
        R_div_C = R / N
        #print(list(R_div_C.size()))
        y = torch.matmul(R_div_C, g_v)  # n_x,n_y,m,d
        y = y.permute(0, 3,2,1).contiguous()  # n_x,n_y,d,m
        y = y.view(batch_size0,self.inter_channels, -1, batch_size1)
        W_y = self.W(y)  # n_x,d,n
        v_star = W_y + v
        v_star = v_star.permute(0, 3,2,1)  # n_x,n_y,m,d

        return v_star




class TxtEncoder(nn.Module):
    def __init__(self, arg):
        super(TxtEncoder, self).__init__()
        self.arg = arg

        # self.device = torch.device("cuda:{}".format(arg.device))

        self.embedding = nn.Embedding(
            num_embeddings=arg.num_embeddings,
            embedding_dim=arg.word_embed_size
        )

        # Loading the GloVe embedding weights
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        if arg.use_glove:
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(arg.pretrained_emb_path)))

        self.lstm = nn.GRU(
            input_size=arg.word_embed_size,
            hidden_size=arg.GRU_hidden,
            num_layers=1,
            batch_first=True,
        )
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

    def forward(self, txt, le):
        txt = self.embedding(txt)
        self.lstm.flatten_parameters()
        packed = pack_padded_sequence(txt, le, batch_first=True)
        out, _ = self.lstm(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True, total_length=self.arg.max_length)
        txt = padded[0]

        return txt

    def _init_lstm(self, weight):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        for w in weight.chunk(3, 0):  # 对张量分块，返回一个张量列表。沿 0轴，分 3块儿
            init.xavier_uniform_(w)


class ImgEncoder(nn.Module):
    def __init__(self, arg):
        super(ImgEncoder, self).__init__()

        self.arg = arg
        self.trans_img = nn.Linear(2048, arg.GRU_hidden)


    def forward(self, img):
        img = self.trans_img(img)

        return img


class Mlp(nn.Module):
    def __init__(self, arg):
        super(Mlp, self).__init__()
        self.arg = arg
        self.l1 = nn.Linear(arg.GRU_hidden, arg.GRU_hidden)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        return x


class Net(nn.Module):
    def __init__(self, arg):
        super(Net, self).__init__()
        self.arg = arg

        self.txtEncoder = TxtEncoder(arg)

        self.imgEncoder = ImgEncoder(arg)

        self.Rs_GCN_1 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_2 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_3 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_4 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_5 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_6 = Rs_GCN(in_channels=512, inter_channels=512)
        self.Rs_GCN_7 = Rs_GCNI(in_channels=512, inter_channels=512)
        self.Rs_GCN_8 = Rs_GCNS(in_channels=512, inter_channels=512)
        self.sat1 = Self_att(arg)
        self.sat2 = Self_att(arg)
        self.linear1 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear2 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.sat3 = Self_att(arg)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.linear_img = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_txt = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_img1 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_txt1 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.dropout1 = nn.Dropout(arg.DROPOUT_R)
        self.dropout2 = nn.Dropout(arg.DROPOUT_R)
        self.rea = Multi_re(arg)

        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        for m in self.modules():
            if isinstance(m, nn.Linear):  # or isinstance(m, nn.Conv2d)
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward_emb(self, img, txt, le):
        txt_feat_mask = self.make_mask(txt.unsqueeze(2))
        img_feat_mask = self.make_mask(img)

        img = self.imgEncoder(img)
        txt = self.txtEncoder(txt, le)

        return img, txt, img_feat_mask, txt_feat_mask

    # img_mask(n_x, 1, 1,n)      txt_mask(n_y, 1,1,m)
    def sim2(self, imgE, txtE, img_mask, txt_mask):
        imgF = imgE
        txtF = txtE
        imgE1=imgE
        txtE1=txtE
        n_x = imgE1.size(0)
        n_y = txtE1.size(0)
        imgE2 = self.Rs_GCN_1(imgE1)
        txtE2 = self.Rs_GCN_2(txtE1)
        imgE3 = self.Rs_GCN_3(imgE2)
        txtE3 = self.Rs_GCN_4(txtE2)
        imgE4 = self.Rs_GCN_5(imgE3)
        txtE4 = self.Rs_GCN_6(txtE3)
        imgE4 = imgE4.unsqueeze(1).expand(-1, n_y, -1, -1)  # (n_x, n_y, n, d)
        txtE4 = txtE4.unsqueeze(0).expand(n_x, -1, -1, -1)  # (n_x, n_y, m, d)
        x_mask = img_mask.expand(-1, n_y, -1, -1)  # n_x,n_y,1,n
        y_mask = txt_mask.squeeze(1).unsqueeze(0).expand(n_x, -1, -1, -1)


        imgD, txtD = self.rea(imgE4, txtE4, x_mask, y_mask)
        imgE,R_aff = self.Rs_GCN_7(imgE4,imgD,txtD)
        txtE = self.Rs_GCN_8(txtE4,imgD,txtD)

        imgE = l2norm(imgE,2)
        txtE= l2norm(txtE,2)
        #s_img = self.sat1(imgF, img_mask.squeeze(2)).squeeze(2)  # (n_x, n)
        #s_txt = self.sat2(txtF, txt_mask.squeeze(2)).squeeze(2)  # (n_y, m)
        #s_img = s_img.unsqueeze(1).expand(-1, n_y, -1)  # (n_x,n_y,n)
        #s_txt = s_txt.unsqueeze(0).expand(n_x, -1, -1)  # (n_x,n_y,m)
        imgE=self.linear_img(imgE)
        txtE=self.linear_txt(txtE)





        sim = torch.matmul(imgE, txtE.transpose(2, 3))  # [n_x, n_y, n, m]
        wx = torch.norm(imgE, 2, 3).unsqueeze(3)
        wy = torch.norm(txtE, 2, 3).unsqueeze(2)
        wxy_norm = torch.matmul(wx, wy).clamp(min=1e-8)
        sim = sim / wxy_norm



        sim_y = sim.masked_fill(y_mask, -1e9)
        sim_x = sim.transpose(2, 3).masked_fill(x_mask, -1e9)
        #print(sim_y.shape)  # 返回张量的size
        #g_f = sim_y[3, 19, 8,6]
        #print(g_f)
        #sim_y = torch.max(sim_y, 3)[0] * s_img  # [n_x, n_y, n]
        #sim_x = torch.max(sim_x, 3)[0] * s_txt  # [n_x, n_y, m]
        sim_y = torch.max(sim_y, 3)[0]   # [n_x, n_y, n]
        #sim_y, indices = torch.sort(-sim_y, dim=2)
        #sim_y = sim_y[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]]
        #sim_y = -sim_y
        sim_x = torch.max(sim_x, 3)[0]   # [n_x, n_y, m]
        sim_x = sim_x.sum(2)
        sim_y = sim_y.sum(2)

        s = sim_y + sim_x*1.5  # (n_x, n_y)
        #score1=s[[0, 1, 2,3,4]]
        #score1=score1[:,[0,1,2]]
        #print("score1")
        #print(score1)
        #print("R_aff")
        #print(R_aff)
        return s


    def forward(self, img, txt, le):  # 直接输出scores矩阵
        imgE, txtE, img_mask, txt_mask = self.forward_emb(img, txt, le)
        # scroes = self.reasoning_sim(img, txt, img_mask, txt_mask)
        scroes = self.sim2(imgE, txtE, img_mask, txt_mask)

        return scroes

    # Masking
    @staticmethod
    def make_mask(feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)