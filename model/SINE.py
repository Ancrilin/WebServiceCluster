import torch
import torch.nn as nn
import torch.nn.functional as F

class Config():
    def __init__(self):
        self.embed = 200                                                 # 节点嵌入维度d
        self.n_vocab = None                                             # 字典数，运行模型时分配
        self.batch_size = 256
        self.epochs = 2
        self.learning_rate = 1e-3                                       # 学习率
        self.dim = 350                                                 # 第二维
        self.delta = 0.5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.alpha = 0.1                                                # 正则化参数
        self.tol = 0.5
        self.vocab = None


class SINE(nn.Module):
    def __init__(self, config):
        super(SINE, self).__init__()
        self.config = config
        self.delta = config.delta
        self.embedding = nn.Embedding(config.n_vocab + 1, config.embed)
        self.layerl1 = nn.Linear(config.embed, config.dim, bias=False)
        self.layerl2 = nn.Linear(config.embed, config.dim, bias=False)
        self.layer2 = nn.Linear(config.dim, 1, bias=False)  # 第二层隐藏层
        self.bias1 = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.register_parameter('bias1', self.bias1)
        # self.register_parameter('bias2', self.bias2)
        self.tanh = nn.Tanh()

    def forward(self, xi, xj, xk):
        xi_emb = self.embedding(xi)
        xj_emb = self.embedding(xj)
        xk_emb = self.embedding(xk)

        zl1 = self.tanh(self.layerl1(xi_emb) + self.layerl2(xj_emb) + self.bias1)
        zl2 = self.tanh(self.layerl1(xi_emb) + self.layerl2(xk_emb) + self.bias1)

        f_pos = self.tanh(self.layer2(zl1) + self.bias2)
        f_neg = self.tanh(self.layer2(zl2) + self.bias2)

        out = F.relu(f_pos - f_neg + self.delta)
        out = torch.sum(out)
        return out

    def regularization_weight(self):                        # l2范数
        regularization_loss = 0
        for param in self.parameters():
            regularization_loss += torch.norm(param, 2)
        return regularization_loss

    # model
    # def _regularizer(self, x):#返回范数
    #     zeros = torch.zeros_like(x)
    #     normed = torch.norm(x - zeros, p=2)#返回输入张量input 的p 范数。
    #     term = torch.pow(normed, 2)
    #     # print('The parameter of ', x)
    #     # print('Yields ',term)
    #     return term
    #
    # def regularize_weights(self):
    #     loss = 0
    #     for parameter in self.parameters():
    #         loss += self._regularizer(parameter)
    #     return loss

    def _get_embedding_id(self, node_id):
        return self.embedding(torch.tensor(node_id)).detach().cpu().numpy()

    def get_embedding(self, name):
        return self._get_embedding_id(torch.tensor(self.config.vocab[name], dtype=torch.long).to(self.config.device))








