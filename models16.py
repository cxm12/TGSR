# coding: utf-8
from model import *


class G_SR_NET_low(nn.Module):  # low frequency SR network
    def __init__(self):
        super(G_SR_NET_low, self).__init__()
        ngf = cfg.GAN.GF_DIM  # 32  feature number
        nef = cfg.TEXT.EMBEDDING_DIM  # 256
        ncf = cfg.GAN.CONDITION_DIM  # 100
        self.ca_net = CA_NET()
        self.h_net1 = INIT_STAGE_GImgup(ngf, ncf, nef)
        self.h_net4 = self.h_net3 = self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
        self.img_net4 = self.img_net3 = self.img_net2 = self.img_net1 = GET_IMAGE_G(ngf)
    
    def forward(self, LR, sent_emb, word_embs, mask):
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        h_code1, att0 = self.h_net1(None, LR, word_embs, mask)
        fake_img1 = self.img_net1(h_code1)  # 32
        fake_imgs.append(fake_img1)
        att_maps.append(att0)
        
        h_code2, att1 = self.h_net2(h_code1, None, word_embs, mask)
        fake_img2 = self.img_net2(h_code2)  # 64
        fake_imgs.append(fake_img2)
        att_maps.append(att1)
        
        h_code3, att2 = self.h_net3(h_code2, None, word_embs, mask)
        fake_img3 = self.img_net3(h_code3)  # 128
        fake_imgs.append(fake_img3)
        att_maps.append(att2)
        
        h_code4, att3 = self.h_net4(h_code3, None, word_embs, mask)
        fake_img4 = self.img_net4(h_code4)  # 256
        fake_imgs.append(fake_img4)
        att_maps.append(att3)
        return fake_imgs, att_maps, mu, logvar


class NetG_high(nn.Module):  # high/low frequency SRResNet
    def __init__(self, cat=False):
        super(NetG_high, self).__init__()
        ngf = cfg.GAN.GF_DIM
        
        self.residual = self.make_layer(ResBlock, 6)
        self.upscale16x = self.upscale8x = self.upscale4x = self.upscale2x = upBlock(ngf, ngf)
        self.cat = cat
        self.conv_output = nn.Sequential(conv5x5(ngf, 3), nn.Tanh())
        
        self.convin = nn.Sequential(conv3x3(3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        self.residual816 = self.residual48 = self.residual24 = nn.Sequential(conv3x3(ngf, ngf * 2),
                                                                             nn.BatchNorm2d(ngf * 2), GLU(),
                                                                             conv3x3(ngf, ngf),
                                                                             nn.BatchNorm2d(ngf))
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_num=32))
        return nn.Sequential(*layers)
    
    def forward(self, LR, SRb, LRb, low=False):
        SRb2 = SRb[0]
        SRb4 = SRb[1]
        SRb8 = SRb[2]
        SRb16 = SRb[3]
        
        if low:
            out = self.convin(LRb)
        else:  # high frequency Net
            out = self.convin(LR - LRb)
        out = self.residual(out)
        #####---------------- UP ------------------###
        out = self.upscale2x(out)
        ims2 = self.conv_output(out) + SRb2
        
        insc2 = out  # torch.cat([out, SRb2], 1) #
        out = self.residual24(insc2)
        out = self.upscale4x(out)
        ims4 = self.conv_output(out) + SRb4
        
        insc4 = out  # torch.cat([out, SRb4], 1) #
        out = self.residual48(insc4)
        out = self.upscale8x(out)
        ims8 = self.conv_output(out) + SRb8
        
        insc8 = out  # torch.cat([out, SRb4], 1) #
        out = self.residual816(insc8)
        out = self.upscale16x(out)
        ims16 = self.conv_output(out) + SRb16
        
        return [ims2, ims4, ims8, ims16]


class NetG_highweight(nn.Module):
    # high/low frequency SRResNet
    def __init__(self, weightmap=False, low='lr-lrblur'):
        super(NetG_highweight, self).__init__()
        ngf = cfg.GAN.GF_DIM
        self.low = low
        
        self.residual = self.make_layer(ResBlock, 6)
        self.upscale4x = upBlock(ngf, ngf)
        self.upscale2x = upBlock(ngf, ngf)
        self.upscale8x = upBlock(ngf, ngf)
        self.upscale16x = upBlock(ngf, ngf)
        self.conv_output = nn.Sequential(conv5x5(ngf, 3), nn.Tanh())
        self.convin = nn.Sequential(conv3x3(3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        self.residual24 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                        nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        self.residual48 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                        nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        self.residual816 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                         nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        
        self.weightmap = weightmap
        if self.weightmap:
            self.a1 = nn.Parameter(torch.ones([32, 32], dtype=torch.float32).cuda())
            self.a2 = nn.Parameter(torch.ones([64, 64], dtype=torch.float32).cuda())
            self.a3 = nn.Parameter(torch.ones([128, 128], dtype=torch.float32).cuda())
            self.a4 = nn.Parameter(torch.ones([256, 256], dtype=torch.float32).cuda())
            self.one1 = self.one2 = self.one3 = self.one4 = torch.ones([1]).cuda()
            print(self.a3.mean(), self.a3.std(), ' = self.a3.mean(), self.a3.std()')
        else:
            self.a = nn.Parameter(torch.FloatTensor([0.5]))
            self.one = torch.ones([1]).cuda()
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_num=32))
        return nn.Sequential(*layers)
    
    def forward(self, LR, SRb, LRb):
        SRb2, SRb4, SRb8, SRb16 = SRb[0], SRb[1], SRb[2], SRb[3]
        
        if self.low == 'lrblur':
            out = self.convin(LRb)
        elif self.low == 'lr-lrblur':
            out = self.convin(LR - LRb)
        elif self.low == 'lr':
            out = self.convin(LR)
        
        out = self.residual(out)
        
        out = self.upscale2x(out)
        if self.weightmap:
            ims2 = self.one1 * self.conv_output(out) + self.a1 * SRb2
        else:
            # self.one = 1 - self.a
            ims2 = self.one * self.conv_output(out) + self.a * SRb2
        
        insc2 = out
        out = self.residual24(insc2)
        out = self.upscale4x(out)
        if self.weightmap:
            ims4 = self.one2 * self.conv_output(out) + self.a2 * SRb4
        else:
            ims4 = self.one * self.conv_output(out) + self.a * SRb4
        
        insc4 = out
        out = self.residual48(insc4)
        out = self.upscale8x(out)
        if self.weightmap:
            ims8 = self.one3 * self.conv_output(out) + self.a3 * SRb8
        else:
            ims8 = self.one * self.conv_output(out) + self.a * SRb8
        
        insc8 = out
        out = self.residual48(insc8)
        out = self.upscale8x(out)
        if self.weightmap:
            ims16 = self.one4 * self.conv_output(out) + self.a4 * SRb16
            return [ims2, ims4, ims8, ims16], self.a4, self.one4
        else:
            ims16 = self.one * self.conv_output(out) + self.a * SRb8
            return [ims2, ims4, ims8, ims16], self.a, self.one
