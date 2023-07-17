import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from GlobalAttention import GlobalAttentionGeneral_weight as ATT_NETw


class DepthToSpace(nn.Module):
    
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(nn.Module):
    
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size
    
    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


# channel halve
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    # return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def conv5x5(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=2, bias=False)
    # return nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=9, stride=1, padding=4, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


def upBlocknoBN(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        GLU())
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num, batchnorm=True):
        super(ResBlock, self).__init__()
        if batchnorm:
            self.block = nn.Sequential(
                conv3x3(channel_num, channel_num * 2),
                nn.BatchNorm2d(channel_num * 2),
                GLU(),
                conv3x3(channel_num, channel_num),
                nn.BatchNorm2d(channel_num))
        else:
            self.block = nn.Sequential(
                conv3x3(channel_num, channel_num * 2),
                GLU(),
                conv3x3(channel_num, channel_num))
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class EDCODER(nn.Module):
    def __init__(self, nef, ngf):
        super(EDCODER, self).__init__()
        self.nef = nef
        self.conv1 = nn.Sequential(
            conv3x3(3, ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            GLU())
        self.d1 = downBlock(ngf, ngf * 2)
        self.d2 = downBlock(ngf * 2, ngf * 4)
        self.d3 = downBlock(ngf * 4, ngf * 8)
        self.u1 = upBlock(ngf * 8, ngf * 4)
        self.conv2 = nn.Sequential(conv3x3(ngf * 8, ngf * 8), nn.BatchNorm2d(ngf * 8), GLU())
        self.u2 = upBlock(ngf * 4, ngf * 2)
        self.conv3 = nn.Sequential(
            conv3x3(ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            GLU())
        self.u3 = upBlock(ngf * 2, ngf)
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x1 = self.conv1(x)  # ngf
        xd1 = self.d1(x1)  # 2ngf
        xd2 = self.d2(xd1)  # 4ngf
        xd3 = self.d3(xd2)  # 8ngf
        xu3 = self.u1(xd3)  # 4ngf
        xd2_2 = torch.cat([xd2, xu3], 1)  # 8ngf
        xd2_3 = self.conv2(xd2_2)  # ngf
        xu2 = self.u2(xd2_3)  # 2ngf
        xd1_2 = torch.cat([xd1, xu2], 1)  # 4ngf
        xd1_3 = self.conv3(xd1_2)  # 2ngf
        xu1 = self.u3(xd1_3)  # 3
        
        out = self.img(xu1)
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions
        
        self.define_module()
        self.init_weights()
    
    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError
    
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())
    
    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps [batch, 18]
        # --> emb: batch x n_steps x ninput  [b, 18, 300]
        emb = self.drop(self.encoder(captions))
        
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()  # [16] --> list长度=batch[]
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)  # 将一个填充过的变长序列压紧。返回一个PackedSequence对象。
        ## embs [241, 256] [18] [None] [None]
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)  ## hidden:[[2, batch, 128], [2, batch, 128]]
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]  # 把压紧的序列再填充回来 [batch, 18, 256]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)  # [batch, 256, 18]
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()  # [batch, 2, 128]
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)  # [batch, 256]
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.nef = nef
        else:
            self.nef = 256  # define a uniform ranker
        
        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)
        
        self.define_module(model)
        self.init_trainable_weights()
    
    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        
        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)
    
    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        
        # image region features
        features = x
        # 17 x 17 x 768
        
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)  # out [batch, 2048, 1, 1]
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)  # out [batch, 2048]
        # 2048
        
        # global image features
        cnn_code = self.emb_cnn_code(x)  # out [batch, 256]
        # 512
        if features is not None:
            features = self.emb_features(features)  # out [batch, 256, 17, 17]
        return features, cnn_code


# ########################   ============================ G networks ========================= ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()
    
    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # 点乘，以e为底的指数
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)  # 1个fully connected layer
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


# 输入noise+sentenceF 输出feature
class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM
        
        self.define_module()
    
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())
        
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
    
    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)  # c_z_code[batch, 200]
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)  # [batch, 8192]
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # [batch, 512, 4, 4]
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)  # [batch, 256, 8, 8]
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)  # [batch, 32, 64, 64]
        
        return out_code64


# 输入 sentenceF 输出feature
class INIT_STAGE_Gup4(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_Gup4, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM
        
        self.define_module()
    
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())
        
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
    
    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)  # c_z_code[batch, 200]
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)  # [batch, 8192]
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # [batch, 512, 4, 4]
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)  # [batch, 256, 8, 8]
        # state size ngf/4 x 16 x 16
        out_code16 = self.upsample2(out_code)
        
        return out_code16


# 输入image+sentenceF 输出feature
class INIT_STAGE_Gim(nn.Module):  # 这样就把c_code用起来了,有全连接层，导致输入大小固定
    def __init__(self, ngf, scale=8, c32=True):
        super(INIT_STAGE_Gim, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM
        self.scale = scale
        if c32:
            self.define_module0()
        else:
            self.define_module()
    
    # attencGAN
    def define_module0(self):
        nz, ngf = self.in_dim, self.gf_dim  # ngf=32
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())  # ngf = 512
        self.upsample1 = upBlock(ngf, ngf)
        self.upsample2 = upBlock(ngf, ngf)
        if self.scale == 8:  # LR = [32]
            self.upsample3 = upBlock(ngf, ngf)  # H,W=64;
        self.fout = nn.Sequential(
            conv3x3(ngf * 2, ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            GLU())
        self.fin = nn.Sequential(
            conv3x3(3, ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            GLU())
    
    # attencGAN1 c_code channel=4
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim  # ngf=32
        ngfim = 32
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())  # ngf = 512
        self.upsample1 = upBlock(ngf, ngf // 2)  # 32-16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)  # 4
        if self.scale == 8:  # LR = [32]
            self.upsample3 = upBlock(ngf // 4, ngf // 8)  # H,W=64; channel=4
        if self.scale == 4:  # LR [64]
            self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.fout = nn.Sequential(
            conv3x3(ngfim + (ngf // 8), ngfim * 2),
            nn.BatchNorm2d(ngfim * 2),
            GLU())
        self.fout1 = nn.Sequential(
            conv3x3(ngfim, ngfim * 2),
            nn.BatchNorm2d(ngfim * 2),
            GLU(),
            conv3x3(ngfim, ngfim),
            nn.BatchNorm2d(ngfim))
        self.fin = nn.Sequential(
            conv3x3(3, ngfim * 2),
            nn.BatchNorm2d(ngfim * 2),
            GLU())
    
    def forward(self, LR, c_code):
        # state size ngf x 4 x 4
        f = self.fin(LR)
        out_code = self.fc(c_code)  # [batch, 16*ngf]  reshape成4*4
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # [batch, 32, 4, 4]
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)  # [batch, 32, 8, 8]
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        if self.scale == 8:  # LR = [32]
            out_code32 = self.upsample3(out_code)
            out_code0 = out_code32
        # state size ngf/16 x 64 x 64
        if self.scale == 4:  # LR = [64]
            out_code32 = self.upsample3(out_code)
            out_code64 = self.upsample4(out_code32)  # [batch, 32, 64, 64]
            out_code0 = out_code64
        out = torch.cat((out_code0, f), 1)
        out1 = self.fout(out)
        # out2 = self.fout1(out1) + f
        
        return out1


# 输入imageF+sentenceF 输出feature
class INIT_STAGE_Gf(nn.Module):  # 这样就把c_code用起来了,有全连接层，导致输入大小固定
    def __init__(self, ngf, scale=8):
        super(INIT_STAGE_Gf, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM
        self.scale = scale
        self.define_module()
    
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim  # ngf=32
        ngfim = 32
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())  # ngf = 512
        self.upsample1 = upBlock(ngf, ngf // 2)  # 32-16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)  # 4
        if self.scale == 8:  # LR = [32]
            self.upsample3 = upBlock(ngf // 4, ngf // 8)  # H,W=64; channel=4
        if self.scale == 4:  # LR [64]
            self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.fout = nn.Sequential(
            conv3x3(ngfim + (ngf // 8), ngfim * 2),
            nn.BatchNorm2d(ngfim * 2),
            GLU())
        # self.fout1 = nn.Sequential(
        #     conv3x3(ngfim, ngfim * 2),
        #     nn.BatchNorm2d(ngfim * 2),
        #     GLU(),
        #     conv3x3(ngfim, ngfim),
        #     nn.BatchNorm2d(ngfim))
    
    def forward(self, LRf, c_code):
        out_code = self.fc(c_code)  # [batch, 16*ngf]  reshape成4*4
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # state size ngf x 4 x 4 # [batch, 32, 4, 4]
        out_code = self.upsample1(out_code)  # state size ngf/3 x 8 x 8 # [batch, 32, 8, 8]
        out_code = self.upsample2(out_code)  # state size ngf/4 x 16 x 16
        if self.scale == 8:  # LR = [32]
            out_code32 = self.upsample3(out_code)  # state size ngf/8 x 32 x 32
            out_code0 = out_code32
        if self.scale == 4:  # LR = [64]
            out_code32 = self.upsample3(out_code)
            out_code64 = self.upsample4(out_code32)  # state size ngf/16 x 64 x 64# [batch, 32, 64, 64]
            out_code0 = out_code64
        out = torch.cat((out_code0, LRf), 1)
        out1 = self.fout(out)
        # out2 = self.fout1(out1) + f
        
        return out1


# 输入sentenceF 输出feature ngf//8 * 32 * 32
class INIT_STAGE_Gfc(nn.Module):
    def __init__(self, ngf, scale=8):
        super(INIT_STAGE_Gfc, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM
        self.scale = scale
        self.define_module()
    
    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim  # ngf=32
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())  # ngf = 512
        self.upsample1 = upBlock(ngf, ngf // 2)  # 32-16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)  # 4
        if self.scale == 8:  # LR = [32]
            self.upsample3 = upBlock(ngf // 4, ngf // 8)  # H,W=64; channel=4
        if self.scale == 1:  # LR = [256]
            self.upsample3 = upBlock(ngf // 4, ngf // 8)  # H,W=64; channel=4
            self.upsample4 = upBlock(ngf // 8, ngf // 8)  #
            self.upsample5 = upBlock(ngf // 8, ngf // 8)  #
            self.upsample6 = upBlock(ngf // 8, ngf // 8)  #
    
    def forward(self, c_code):
        out_code = self.fc(c_code)  # [batch, 16*ngf]  reshape成4*4
        out_code = out_code.view(-1, self.gf_dim, 4, 4)  # state size ngf x 4 x 4 # [batch, 32, 4, 4]
        out_code = self.upsample1(out_code)  # state size ngf/3 x 8 x 8 # [batch, 32, 8, 8]
        out_code = self.upsample2(out_code)  # state size ngf/4 x 16 x 16
        if self.scale == 8:  # LR = [32]
            out_code32 = self.upsample3(out_code)  # state size ngf/8 x 32 x 32
            out_code0 = out_code32
        if self.scale == 4:  # LR = [64]
            out_code32 = self.upsample3(out_code)
            out_code64 = self.upsample4(out_code32)  # state size ngf/16 x 64 x 64# [batch, 32, 64, 64]
            out_code0 = out_code64
        if self.scale == 1:  # LR = [64]
            out_code32 = self.upsample3(out_code)
            out_code64 = self.upsample4(out_code32)  # state size ngf/16 x 64 x 64# [batch, 32, 64, 64]
            out_code128 = self.upsample5(out_code64)  # state size ngf/16 x 64 x 64# [batch, 32, 64, 64]
            out_code256 = self.upsample6(out_code128)  # state size ngf/16 x 64 x 64# [batch, 32, 64, 64]
            out_code0 = out_code256
        return out_code0


# 输入image+ wordF 输出feature
class INIT_STAGE_GImg(nn.Module):  # 忽视了c_code
    def __init__(self, ngf, ncf, nef, weightatten=False):
        super(INIT_STAGE_GImg, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf  # cfg.TEXT.EMBEDDING_DIM   # 100+100
        self.ef_dim = nef
        self.weightattn = weightatten  # 对wordF赋予新权重
        self.define_module()
    
    def define_module(self):
        if self.weightattn:
            self.att = ATT_NETw(self.gf_dim, self.ef_dim)
        else:
            self.att = ATT_NET(self.gf_dim, self.ef_dim)
        nz, ngf = self.in_dim, self.gf_dim
        self.im2f = nn.Sequential(  # nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(3, self.gf_dim * 2),
            nn.BatchNorm2d(self.gf_dim * 2),
            GLU())
        self.fout = nn.Sequential(
            conv3x3(self.gf_dim * 2, self.gf_dim * 2),
            nn.BatchNorm2d(self.gf_dim * 2),
            GLU())
        
        self.residual = self._make_layer(ResBlock, ngf * 2)
    
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    
    def forward(self, z_code0, c_code0, LR, word_embs, mask):  # input [batchsize, 100]  out [batchsize,32,64,64]
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        # c_z_code = torch.cat((c_code, z_code), 1) # [bs,200]
        # # state size ngf x 4 x 4
        # out_code = self.fc(c_z_code) # [bs, 8192 = 256*4*4*2]
        # out_code = out_code.view(-1, self.gf_dim, 4, 4) # [bs, 512,4,4]
        # # state size ngf/3 x 8 x 8
        # out_code = self.upsample1(out_code)
        # # state size ngf/4 x 16 x 16
        # out_code = self.upsample2(out_code)
        # # state size ngf/8 x 32 x 32
        # out_code32 = self.upsample3(out_code)
        # # state size ngf/16 x 64 x 64
        # out_code64 = self.upsample4(out_code32) # []
        self.att.applyMask(mask)  # [bs,18]
        h_code = self.im2f(LR)
        c_code, att = self.att(h_code, word_embs)  # LR[bs,3,128,128] = h_code:[bs,32=ngf,128,128]
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code1 = self.residual(h_c_code)
        out_code = self.fout(out_code1)
        # state size ngf/2 x 2in_size x 2in_size
        # out_code = self.upsample(out_code)
        
        return out_code  # 64


# 输入image+wordF 输出x2 Up feature
class INIT_STAGE_GImgup(nn.Module):
    def __init__(self, ngf, ncf, nef, batchnorm=True):
        super(INIT_STAGE_GImgup, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.GAN.Z_DIM + ncf
        self.ef_dim = nef
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.define_module()
        else:
            self.define_modulenoBN()
    
    def define_module(self):
        self.att = ATT_NET(self.gf_dim, self.ef_dim)
        ngf = self.gf_dim
        self.im2f = nn.Sequential(
            conv3x3(3, self.gf_dim * 2),
            nn.BatchNorm2d(self.gf_dim * 2),
            GLU())
        self.upsample = upBlock(ngf * 2, ngf)
        self.residual = self._make_layer(ResBlock, ngf * 2)
    
    def define_modulenoBN(self):
        self.att = ATT_NET(self.gf_dim, self.ef_dim)
        ngf = self.gf_dim
        self.im2f = nn.Sequential(
            conv3x3(3, self.gf_dim * 2),
            GLU())
        self.upsample = upBlocknoBN(ngf * 2, ngf)
        self.residual = self._make_layer(ResBlock, ngf * 2, BN=False)
    
    def _make_layer(self, block, channel_num, BN=True):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num, BN))
        return nn.Sequential(*layers)
    
    def forward(self, c_code0, LR, word_embs, mask):  # input [batchsize, 100]  out [batchsize,32,64,64]
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x ngf/16 x 64 x 64
        """
        self.att.applyMask(mask)  # [bs,18]
        h_code = self.im2f(LR)
        # h_code = LR
        c_code, att = self.att(h_code, word_embs)  # LR[bs,3,128,128] = h_code:[bs,32=ngf,128,128]
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code1 = self.residual(h_c_code)
        
        out_code = self.upsample(out_code1)
        return out_code, att


# 输入feature+wordF 输出x2 Up feature
class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, weightatten=False):
        super(NEXT_STAGE_G, self).__init__()
        self.weightattn = weightatten  # 对wordF赋予新权重
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM  # 2
        self.define_module()
    
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    
    def define_module(self):
        ngf = self.gf_dim
        if self.weightattn:
            self.att = ATT_NETw(ngf, self.ef_dim)
            self.att = ATT_NETw(ngf, self.ef_dim)
        else:
            self.att = ATT_NET(ngf, self.ef_dim)  # input: ef_dim
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)
    
    def forward(self, h_code, c_code0, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        # word_embs [batch, 256, 18]
        c_code, att = self.att(h_code, word_embs)  # [bs,32,128,128], [bs, 18, 128,128]
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        
        return out_code, att


# 输入feature 输出x2 Up feature 没有Text-Attention
class NEXT_STAGE_G_noAttn(nn.Module):
    def __init__(self, ngf, nef):
        super(NEXT_STAGE_G_noAttn, self).__init__()
        self.gf_dim = ngf  # 32
        self.ef_dim = nef  # 256
        self.num_residual = cfg.GAN.R_NUM  # 2
        self.define_module()
    
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    
    def define_module(self):
        ngf = self.gf_dim
        self.noatt = nn.Sequential(
            conv3x3(ngf, ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            GLU())
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf)
    
    def forward(self, h_code):
        c_code = self.noatt(h_code)
        out_code = self.residual(c_code)
        
        out_code = self.upsample(out_code)
        
        return out_code


# 输入feature+wordF 输出 feature
class NEXT_STAGE_G_LR(nn.Module):  # no upsample
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G_LR, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()
    
    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)
    
    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.fout = nn.Sequential(conv3x3(ngf * 2, ngf * 2),
                                  nn.InstanceNorm2d(ngf * 2), GLU())
        # nn.BatchNorm2d(ngf * 2), GLU())
    
    def forward(self, h_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)  # [bs,32,128,128], [bs, 18, 128,128];
        # word_embs: batch x 256 x seq_len
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)
        out_code1 = self.fout(out_code)
        return out_code1, att


# 一个conv生成3通道image
class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )
    
    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


## No activation
class GET_IMAGE_G_noAct(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G_noAct, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3)
        )
    
    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class GET_IMAGE_G_Bic(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G_Bic, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )
    
    def forward(self, h_code, bic):
        out_img = self.img(h_code) + bic
        return out_img


###Image-Adaptive Word Demand
class IAWD(nn.Module):
    def __init__(self, ngf):
        super(IAWD, self).__init__()
        self.conv1 = conv3x3(ngf, 256)
        self.conv2 = conv3x3(ngf, 256)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, imf, wf):
        batch_size, ih, iw = imf.size(0), imf.size(2), imf.size(3)
        queryL = ih * iw
        h1 = self.conv1(imf)  # out: [b, 256, 256, 256]
        h2 = self.conv2(imf)
        
        h1_1 = h1.view(batch_size, -1, queryL)  # b,256,h*w
        h2_1 = h2.view(batch_size, -1, queryL)  # b,256,h*w
        
        h2_1T = torch.transpose(h2_1, 1, 2)  # b,h*w,256
        h4 = self.sm(torch.bmm(h1_1, h2_1T)).view(batch_size, 256, 256)  # b, 256, 256
        
        weight = h4  # # b, 256, 256
        IAWF = torch.bmm(weight, wf)  # wf=[batch, 256, 18]
        IAWF = IAWF + wf
        
        return IAWF


class Word_atten(nn.Module):  # 类似channel attention的对每个word有个weight; 不限制输入feature的H、W
    def __init__(self, ngf, outf=256):
        super(Word_atten, self).__init__()
        self.outf = outf
        self.conv1 = conv3x3(ngf, outf)
        self.conv2 = conv3x3(ngf, outf)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, imf, wf):
        batch_size, ih, iw = imf.size(0), imf.size(2), imf.size(3)
        queryL = ih * iw
        
        h1 = self.conv1(imf)  # out: [b, 256, H, W]
        h2 = self.conv2(imf)
        h1_1 = h1.view(batch_size, -1, queryL)  # b,256,h*w
        h2_1 = h2.view(batch_size, -1, queryL)  # b,256,h*w
        
        h2_1T = torch.transpose(h2_1, 1, 2)  # b,h*w,256
        weight = self.sm(torch.bmm(h1_1, h2_1T)).view(batch_size, self.outf, self.outf)  # b, 256, 256
        
        IAWF = self.sm(torch.bmm(weight, wf))  # IAWF = wf = [batch, 256, 14]
        wei = nn.AdaptiveAvgPool2d((1, wf.size(2)))(IAWF)  # [b, 1, 14]
        IAWF = wei * wf  # [b, 1, 14]* [b, 256, 14] -> [b, 256, 14]
        
        return IAWF


class IAWDsent(nn.Module):
    def __init__(self, ngf):
        super(IAWDsent, self).__init__()
        self.conv1 = conv3x3(ngf, 1)
        self.conv2 = conv3x3(ngf, 1)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, imf, wf):
        batch_size, ih, iw = imf.size(0), imf.size(2), imf.size(3)
        queryL = ih * iw
        
        h1 = self.conv1(imf)  # out: [b, 1, 256, 256]
        h2 = self.conv2(imf)
        
        h1 = h1.view(batch_size, -1, queryL)  # b,1,h*w
        h2 = h2.view(batch_size, -1, queryL)  # b,1,h*w
        
        h2T = torch.transpose(h2, 1, 2)  # b,h*w,1
        h4 = self.sm(torch.bmm(h2T, h1)).view(batch_size, 1, queryL, queryL)  # b, hw, hw
        
        weight = nn.AvgPool2d(kernel_size=5, stride=4, padding=2)(h4)  # b, hw/4, hw/4
        wf = wf.unsqueeze(2)
        IAWF = torch.bmm(weight.squeeze(1), wf)  # wf=[batch, 256, 14]
        IAWF = IAWF + wf
        IAWF = IAWF.squeeze(2)
        
        return IAWF


class IAWDword(nn.Module):
    def __init__(self, ngf):
        super(IAWDword, self).__init__()
        self.conv1 = conv3x3(ngf, 1)
        self.conv2 = conv3x3(ngf, 1)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, imf, wf):
        batch_size, ih, iw = imf.size(0), imf.size(2), imf.size(3)
        queryL = ih * iw
        
        h1 = self.conv1(imf)  # out: [b, 1, 256, 256]
        h2 = self.conv2(imf)
        
        h1 = h1.view(batch_size, -1, queryL)  # b,1,h*w
        h2 = h2.view(batch_size, -1, queryL)  # b,1,h*w
        
        h2T = torch.transpose(h2, 1, 2)  # b,h*w,1
        h4 = self.sm(torch.bmm(h2T, h1)).view(batch_size, 1, queryL, queryL)  # b, hw, hw
        
        weight = nn.AvgPool2d(kernel_size=5, stride=4, padding=2)(h4)  # b, hw/4, hw/4
        IAWF = torch.bmm(weight.squeeze(1), wf)  # wf=[batch, 256, 14]
        IAWF = IAWF + wf
        
        return IAWF


class IAWDspatial(nn.Module):  # spatial attention
    def __init__(self, ngf):
        super(IAWDspatial, self).__init__()
    
    def forward(self, imf, wf):
        batch_size, ih, iw = imf.size(0), imf.size(2), imf.size(3)
        queryL = ih * iw
        hs = torch.sum(imf, 1)  # 求和
        hs1 = hs.view(batch_size, -1, queryL)  # reshape
        hssm = self.sm(hs1)  # softmax
        weight = hssm.view(batch_size, ih, iw)  # reshape
        
        IAWF = torch.bmm(weight, wf)  # wf=[batch, 256, 14]
        IAWF = IAWF + wf
        
        return IAWF
