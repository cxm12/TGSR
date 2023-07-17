# coding: utf-8
import torch.nn.parallel
from util import *

useBN = False


class _Residual_Block(nn.Module):
    def __init__(self, norm=False, ngf=64):
        super(_Residual_Block, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=1, bias=False)
        if norm:
            # self.in1 = nn.InstanceNorm2d(ngf, affine=True) #
            # self.in2 = nn.InstanceNorm2d(ngf, affine=True) #
            self.in1 = nn.BatchNorm2d(ngf)  #
            self.in2 = nn.BatchNorm2d(ngf)  #
    
    def forward(self, x):
        identity_data = x
        if self.norm:
            output = self.relu(self.in1(self.conv1(x)))
            output = self.in2(self.conv2(output))
        else:
            output = self.relu(self.conv1(x))
            output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


############### ------------------- Low High frequent separate ---------------############
class G_SR_NET_low(nn.Module):  # low frequency SR network
    def __init__(self):
        super(G_SR_NET_low, self).__init__()
        ngf = cfg.GAN.GF_DIM  # 32  feature number
        nef = cfg.TEXT.EMBEDDING_DIM  # 256
        ncf = cfg.GAN.CONDITION_DIM  # 100
        self.ca_net = CA_NET()
        self.h_net1 = INIT_STAGE_GImgup(ngf, ncf, nef)
        self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
        self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
        self.img_net1 = GET_IMAGE_G_noAct(ngf)  # GET_IMAGE_G(ngf)  #
        self.img_net2 = GET_IMAGE_G_noAct(ngf)  # GET_IMAGE_G(ngf)  #
        self.img_net3 = GET_IMAGE_G_noAct(ngf)  # GET_IMAGE_G(ngf)  #
    
    def forward(self, LR, sent_emb, word_embs, mask, outmiddle=False):
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        h_code1, att0 = self.h_net1(None, LR, word_embs, mask)
        ft1 = h_code1
        #
        fake_img1 = self.img_net1(h_code1)  # out [batchsize, 3, 64]
        fake_imgs.append(fake_img1)
        if att0 is not None:
            att_maps.append(att0)
        
        h_code2, att1 = self.h_net2(h_code1, None, word_embs, mask)
        ft2 = h_code2
        fake_img2 = self.img_net2(h_code2)  # [batchsize, 3, 128]
        fake_imgs.append(fake_img2)
        if att1 is not None:
            att_maps.append(att1)
        
        h_code3, att2 = self.h_net3(h_code2, None, word_embs,
                                    mask)  # output: [batchsize,32,256,256]  [batchsize,18,128,128]
        ft3 = h_code3
        fake_img3 = self.img_net3(h_code3)
        fake_imgs.append(fake_img3)
        if att2 is not None:
            att_maps.append(att2)
        
        if outmiddle:
            return fake_imgs, att_maps, mu, logvar, [ft1, ft2, ft3]
        else:
            return fake_imgs, att_maps, mu, logvar


class G_SR_NET_low_stage1(nn.Module):
    # low frequency SR network
    def __init__(self):
        super(G_SR_NET_low_stage1, self).__init__()
        ngf = cfg.GAN.GF_DIM  # 32  feature number
        nef = cfg.TEXT.EMBEDDING_DIM  # 256
        ncf = cfg.GAN.CONDITION_DIM  # 100
        self.ca_net = CA_NET()
        self.up1 = nn.Sequential(
            conv3x3(ngf, ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            upBlocknoBN(ngf * 2, ngf))
        #
        self.h_net1 = INIT_STAGE_GImgup(ngf, ncf, nef, batchnorm=False)
        self.img_net1 = GET_IMAGE_G(ngf)
        self.up2 = nn.Sequential(
            conv3x3(ngf, ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            upBlocknoBN(ngf * 2, ngf))
        
        self.h_net2 = ResBlock(ngf, batchnorm=False)
        self.img_net2 = GET_IMAGE_G(ngf)
        self.h_net3 = ResBlock(ngf, batchnorm=False)
        self.img_net3 = GET_IMAGE_G(ngf)
    
    def forward(self, LR, sent_emb, word_embs, mask):
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)
        h_code1, attn0 = self.h_net1(None, LR, word_embs, mask)
        #
        fake_img1 = self.img_net1(h_code1)  # out [batchsize, 3, 64]
        fake_imgs.append(fake_img1)
        att_maps.append(attn0)
        
        h_code1 = self.up1(h_code1)
        h_code2 = self.h_net2(h_code1)
        fake_img2 = self.img_net2(h_code2)  # [batchsize, 3, 128]
        fake_imgs.append(fake_img2)
        attn1 = nn.Upsample(scale_factor=2, mode='nearest')(attn0)
        att_maps.append(attn1)
        
        h_code2 = self.up1(h_code2)
        h_code3 = self.h_net3(h_code2)  # output: [batchsize,32,256,256]  [batchsize,18,128,128]
        fake_img3 = self.img_net3(h_code3)
        fake_imgs.append(fake_img3)
        attn2 = nn.Upsample(scale_factor=2, mode='nearest')(attn1)
        att_maps.append(attn2)
        
        return fake_imgs, att_maps, mu, logvar


class NetG_high(nn.Module):  # high/low frequency SRResNet
    def __init__(self, cat=False):
        super(NetG_high, self).__init__()
        ngf = cfg.GAN.GF_DIM
        # self.conv_mid = nn.Sequential(conv3x3(ngf * 2, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        
        self.residual = self.make_layer(ResBlock, 6)
        self.upscale4x = upBlock(ngf, ngf)
        self.upscale2x = upBlock(ngf, ngf)
        self.upscale8x = upBlock(ngf, ngf)
        self.cat = cat
        self.conv_output = nn.Sequential(conv5x5(ngf, 3), nn.Tanh())
        if not cat:
            self.convin = nn.Sequential(conv3x3(3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
            self.residual24 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                            nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
            self.residual48 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                            nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        else:
            self.convin = nn.Sequential(conv3x3(2 * 3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
            self.conv_output8 = nn.Sequential(conv5x5(2 * 3, 3), nn.Tanh())
            self.residual24 = nn.Sequential(conv3x3(2 * 3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                            nn.BatchNorm2d(ngf))
            self.residual48 = nn.Sequential(conv3x3(2 * 3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                            nn.BatchNorm2d(ngf))
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_num=32))
        return nn.Sequential(*layers)
    
    def forward(self, LR, SRb, LRb, low=False, outmiddle=False):
        SRb2 = SRb[0]
        SRb4 = SRb[1]
        SRb8 = SRb[2]
        
        if self.cat:  # concatenation
            LRh = LR - LRb
            out = self.convin(torch.cat([LRh, LRb], 1))
            out = self.upscale2x(out)
            ims2 = self.conv_output(out)
            
            out = self.residual24(torch.cat([ims2, SRb2], 1))
            out = self.upscale4x(out)
            ims4 = self.conv_output(out)
            
            out = self.residual48(torch.cat([ims4, SRb4], 1))
            out = self.upscale8x(out)
            ims8 = self.conv_output(out)
            # ims8 = self.conv_output8(torch.cat([ims8, SRb8], 1))
        else:
            if low:  # 低low frequency network的输入为低频LR: SRb = Bicb  # cat=False
                out = self.convin(LRb)
            else:  # high frequency Net
                out = self.convin(LR - LRb)
            out = self.residual(out)
            #####---------------- UP ------------------###
            out = self.upscale2x(out)
            f1 = out
            ims2 = self.conv_output(out) + SRb2
            
            insc2 = out  # torch.cat([out, SRb2], 1) #
            out = self.residual24(insc2)
            out = self.upscale4x(out)
            f2 = out
            ims4 = self.conv_output(out) + SRb4
            
            insc4 = out  # torch.cat([out, SRb4], 1) #
            out = self.residual48(insc4)
            out = self.upscale8x(out)
            f3 = out
            ims8 = self.conv_output(out) + SRb8
        if outmiddle:
            return [ims2, ims4, ims8], [f1, f2, f3]
        else:
            return [ims2, ims4, ims8]


class NetG_highweight(nn.Module):
    # high/low frequency SRResNet  # 加权结合High/Low网络输出
    def __init__(self, weightmap=False, low='lr-lrblur', useAct=True):
        super(NetG_highweight, self).__init__()
        ngf = cfg.GAN.GF_DIM
        self.low = low
        
        self.residual = self.make_layer(ResBlock, 6)
        self.upscale4x = upBlock(ngf, ngf)
        self.upscale2x = upBlock(ngf, ngf)
        self.upscale8x = upBlock(ngf, ngf)
        if useAct:
            self.conv_output = nn.Sequential(conv5x5(ngf, 3), nn.Tanh())
        else:
            self.conv_output = nn.Sequential(conv5x5(ngf, 3))
        
        self.convin = nn.Sequential(conv3x3(3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        self.residual24 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                        nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        self.residual48 = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(), conv3x3(ngf, ngf),
                                        nn.BatchNorm2d(ngf))  # self.make_layer(ResBlock, 1)
        
        self.weightmap = weightmap
        if self.weightmap:  # self.a = nn.Parameter(torch.FloatTensor(256, 256)) # 初始化以0为均值
            self.a1 = nn.Parameter(torch.ones([64, 64],
                                              dtype=torch.float32).cuda())  # nn.Parameter(torch.randn([64, 64], dtype=torch.float32).cuda() + 0.5)
            self.a2 = nn.Parameter(torch.ones([128, 128], dtype=torch.float32).cuda())
            self.a3 = nn.Parameter(torch.ones([256, 256], dtype=torch.float32).cuda())
            self.one1 = self.one2 = self.one3 = torch.ones([1]).cuda()
            # self.one1 = (1 - self.a1)
            # self.one2 = (1 - self.a2)
            # self.one3 = (1 - self.a3)
            print(self.a3.mean(), self.a3.std(), ' = self.a3.mean(), self.a3.std()')
        else:
            self.a = nn.Parameter(
                torch.FloatTensor([0.5])).cuda()  # # 将不可训练类型Tensor转换成可训练类型parameter并将这parameter绑定module里
            self.one = torch.ones([1]).cuda()  # 418
        # self.a = torch.FloatTensor([0.5]).cuda()
        # self.a.requires_grad = True
        # self.one = torch.ones((1), dtype=torch.float32).cuda()
        # a = Variable(torch.Tensor([2]), requires_grad=True)
        # a = tf.convert_to_tensor(tf.Variable(initial_value=[1], dtype=tf.float32, trainable=True, name="weight"),dtype=tf.float32, name='weighta')
        # a = torch.tensor([1]) # value=1
        # tf.Variable(initial_value=[1], trainable=False, name="weight1") #
        # self.bias = Parameter(torch.Tensor(out_channels))
    
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channel_num=32))
        return nn.Sequential(*layers)
    
    def forward(self, LR, SRb, LRb):
        SRb2, SRb4, SRb8 = SRb[0], SRb[1], SRb[2]
        # SRb2 = torch.round(SRb2 * 255.0)/255.0
        if self.low == 'lrblur':  # 直接输入LR
            out = self.convin(LRb)
        elif self.low == 'lr-lrblur':  # 输入高频LR
            out = self.convin(LR - LRb)
        elif self.low == 'lr':  # 输入LR
            out = self.convin(LR)
        out = self.residual(out)
        #####---------------- UP ------------------###
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
            return [ims2, ims4, ims8], self.a3, self.one3
        else:
            ims8 = self.one * self.conv_output(out) + self.a * SRb8
            return [ims2, ims4, ims8], self.a, self.one


class NetG_high_SR_weight(nn.Module):
    def __init__(self):
        super(NetG_high_SR_weight, self).__init__()
        ngf = cfg.GAN.GF_DIM
        self.down = SpaceToDepth(block_size=4)
        self.upscale = DepthToSpace(block_size=4)
        self.upscale2x = upBlock(ngf, ngf)
        self.conv_output = nn.Sequential(conv5x5(ngf, 3), nn.Tanh())
        self.convin1 = nn.Sequential(conv3x3(3, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        self.convin = nn.Sequential(conv3x3(3 * 4 * 4, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU())
        self.convmd = nn.Sequential(conv3x3(ngf * 2, ngf * 4 * 4 * 2), nn.BatchNorm2d(ngf * 4 * 4 * 2), GLU())
        self.residual1 = self.residual = nn.Sequential(conv3x3(ngf, ngf * 2), nn.BatchNorm2d(ngf * 2), GLU(),
                                                       conv3x3(ngf, ngf),
                                                       nn.BatchNorm2d(ngf))
        self.weightmap = True
        if self.weightmap:
            self.a = nn.Parameter(torch.ones([256, 256], dtype=torch.float32).cuda())
        else:
            self.a = nn.Parameter(torch.FloatTensor([0.5]))
        print(self.a.mean(), self.a.std(), ' = self.a.mean(), self.a.std()')
    
    def forward(self, LR, SRb, LRb):
        out = self.convin1(LR)  # - LRb
        out = self.residual1(out)
        xlr = self.upscale2x(out)  ##--UP
        
        SR = SRb[-1]
        x = self.down(SR)
        out = self.convin(x)
        out = self.residual(out)
        
        out1 = torch.cat([xlr, out], 1)
        out2 = self.convmd(out1)
        out2 = self.upscale(out2)
        im = self.conv_output(out2) + self.a * SR
        return [im], self.a
