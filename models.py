import torch.nn as nn
import torch,math

import torch.nn.functional as F
import numpy as np


from net_utils import FC, MLP, LayerNorm,l2norm,cosine_similarity_a2a

# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, arg):
        super(MHAtt, self).__init__()
        self.arg = arg

        self.linear_v = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_k = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_q = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)

        self.dropout = nn.Dropout(arg.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.arg.MULTI_HEAD,
            self.arg.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.arg.MULTI_HEAD,
            self.arg.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.arg.MULTI_HEAD,
            self.arg.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.arg.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)   # 将 mask中为 1的索引替换为value

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, arg):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=arg.HIDDEN_SIZE,
            mid_size=arg.FF_SIZE,
            out_size=arg.HIDDEN_SIZE,
            dropout_r=arg.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, arg):
        super(SA, self).__init__()

        self.mhatt = MHAtt(arg)
        self.ffn = FFN(arg)

        self.dropout1 = nn.Dropout(arg.DROPOUT_R)
        self.norm1 = LayerNorm(arg.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(arg.DROPOUT_R)
        self.norm2 = LayerNorm(arg.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


class Self_att(nn.Module):
    def __init__(self, arg):
        super(Self_att, self).__init__()

        # self.mlp = MLP(
        #     in_size=arg.GRU_hidden,
        #     mid_size=int(arg.GRU_hidden/8),
        #     out_size=1,
        #     dropout_r=0.1,
        #     use_relu=True,
        # )
        self.mlp = nn.Linear(arg.GRU_hidden,1)


    def forward(self, x, txt_mask):  # [batch, n, GRU_hidden]
        txt_mask = txt_mask.squeeze(2).squeeze(1)
        att = self.mlp(x)

        b, n, _ = x.size()
        att = att.view(b, n).masked_fill(txt_mask, -1e9)
        att = F.softmax(att, dim=1).view(b, n, 1)
        x = x * att
        #txt_mask = (txt_mask.squeeze(2).squeeze(1) == 0).float()
        #x = (x * txt_mask.unsqueeze(2)).sum(1) / txt_mask.sum(1).unsqueeze(1)
        x = torch.sum(x, dim=1)

        return x  # [batch, GRU_hidden]



class Fuse(nn.Module):
    def __init__(self, arg):
        super(Fuse, self).__init__()

        self.linear_img = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_txt = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_img1 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)
        self.linear_txt1 = nn.Linear(arg.HIDDEN_SIZE, arg.HIDDEN_SIZE)


        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

        self.dropout1 = nn.Dropout(arg.DROPOUT_R)
        self.dropout2 = nn.Dropout(arg.DROPOUT_R)


    def forward(self, imgE, txtE, x_mask, y_mask):
        x = self.linear_img(imgE)
        y = self.linear_txt(txtE)
        imgE1=imgE
        txtE1=txtE
        att = torch.matmul(x, y.transpose(2, 3))
        wx = torch.norm(x, 2, 3).unsqueeze(3)
        wy = torch.norm(y, 2, 3).unsqueeze(2)
        wxy_norm = torch.matmul(wx, wy).clamp(min=1e-8)  # [n, m]
        att = att / wxy_norm  # [n_x, n_y, n, m]
        att1 = att.masked_fill(y_mask, -1e9)
        att1 = F.softmax(att1, dim=3)
        att1 = self.dropout1(att1)
        att_x = torch.matmul(att1, txtE1)
        gate_x = self.sig1(att_x * imgE1)
        fuse_x = self.linear_img1(gate_x * att_x)
        imgE = fuse_x + imgE1

        att2 = att.transpose(2, 3).masked_fill(x_mask, -1e9)
        att2 = F.softmax(att2, dim=3)  # [m,n]
        att2 = self.dropout2(att2)
        att_y = torch.matmul(att2, imgE1)
        gate_y = self.sig2(att_y * txtE1)
        fuse_y = self.linear_txt1(gate_y * att_y)
        txtE = fuse_y + txtE1


        return imgE,txtE

        # similarity = cosine_similarity_a2a(x, y)
        #
        # fuse_x = []
        # fuse_y = []
        #
        # for i in range(similarity.size(0)):
        #     sim = similarity[i]   # (n, m)
        #     xi = x[i]
        #     yi = y[i]
        #
        #     sim_x = sim.masked_fill(y_mask[0].squeeze(0), -1e9)    # 每行为一个x和每个y的相似的
        #     sim_y = sim.transpose(0,1).masked_fill(x_mask[0].squeeze(0), -1e9)
        #
        #     id_x = torch.max(sim_x, 1)[1]   # 给每个x，找出最相似y的索引
        #     id_y = torch.max(sim_y, 1)[1]
        #
        #     a, b = [], []
        #     for _, j in enumerate(id_x):
        #         a.append(yi[j].unsqueeze(0))
        #     for _, k in enumerate(id_y):
        #         b.append(xi[k].unsqueeze(0))
        #
        #     x_max_y = torch.cat(a, 0)  # 最相似的y，对每个x
        #     y_max_x = torch.cat(b, 0)
        #
        #     gate_x_b = self.sig1(xi*x_max_y)
        #     gate_y_b = self.sig2(yi*y_max_x)
        #
        #     gate_x_w = self.sig3(self.fuse_gate_img(torch.cat((xi,x_max_y), 1)))
        #     gate_y_w = self.sig4(self.fuse_gete_txt(torch.cat((yi,y_max_x), 1)))
        #
        #     fuse1 = (xi + x_max_y) * gate_x_w + gate_x_b
        #     fuse2 = (yi + y_max_x) * gate_y_w + gate_y_b
        #
        #     fuse_x.append(fuse1.unsqueeze(0))
        #     fuse_y.append(fuse2.unsqueeze(0))
        #
        # return torch.cat(fuse_x, 0), torch.cat(fuse_y, 0)


class Cell(nn.Module):
    def __init__(self, arg):
        super(Cell, self).__init__()

        self.reasoning = Fuse(arg)

    def forward(self, x, y, x_mask, y_mask):
        res_x = x     # img
        res_y = y     # txt

        x, y = self.reasoning(x, y, x_mask, y_mask)

        #return x + res_x, y + res_y
        return x, y



class Multi_re(nn.Module):
    def __init__(self, arg):
        super(Multi_re, self).__init__()

        self.res_blocks = nn.ModuleList()

        for _ in range(arg.n_res_blocks):
            self.res_blocks.append(Fuse(arg))


    def forward(self, x, y, x_mask, y_mask):
        for i, res_block in enumerate(self.res_blocks):

            x, y = res_block(x, y, x_mask, y_mask)

        return x, y


















