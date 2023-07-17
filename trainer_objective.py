# coding: utf-8
from __future__ import print_function
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.utils import mkdir_p, build_super_imagesall
from model import G_SR_NET_low_stage1, RNN_ENCODER, Variable, torch, cfg
from datasets import prepare_datablur

import os
import numpy as np


def to_np(x):
    return x.cpu().data.numpy()


class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword, cfg):
        self.cfg = cfg
        if cfg.TRAIN.FLAG:
            print('output dir during training', output_dir)
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.logdir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.logdir)
        
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True
        
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        
        self.n_words = n_words
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
    
    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # 全1向量tensor
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))  # 全0向量tensor
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if self.cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()
        
        return real_labels, fake_labels, match_labels
    
    def gen_exampleSRHL(self, savefile='changetxt'):
        stage1 = False  # True  #
        input_NetGH = 'lr'  # 'lr-lrblur'  # input of high-frequency net is LR or blurred_LR
        weightmap = False  # True  # high-frequency net combine low-frequency SR image by a weight map
        
        text_batch_num = 100  # number of test image
        # Build and load the generator
        text_encoder = RNN_ENCODER(self.n_words, nhidden=self.cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(self.cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        print('Load text encoder from:', self.cfg.TRAIN.NET_E)
        text_encoder = text_encoder.cuda()
        text_encoder.eval()
        
        # the path to save generated images
        print('G Net')
        s_tmp = self.cfg.TRAIN.NET_G[:self.cfg.TRAIN.NET_G.rfind('.pth')]
        
        ## ------------- low frequent ---------------- ##
        if cfg.TREE.BRANCH_NUM == 4:
            from model import G_SR_NET_low
            if stage1:
                netGL = G_SR_NET_low_stage1()
            else:
                netGL = G_SR_NET_low()
        else:
            from models16 import G_SR_NET_low
            netGL = G_SR_NET_low()
        
        if cfg.TREE.BRANCH_NUM == 4:
            from model import NetG_highweight
        else:
            from models16 import NetG_highweight
        netGH = NetG_highweight(weightmap=weightmap, low=input_NetGH)
        
        if self.cfg.TRAIN.NET_G != '':
            netGL.load_state_dict(torch.load(self.cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage))
            netGH.load_state_dict(
                torch.load(self.cfg.TRAIN.NET_G.replace('netG', 'netGH'), map_location=lambda storage, loc: storage))
            print(' Load G from: ', self.cfg.TRAIN.NET_G)
        
        netGL.cuda()
        netGH.cuda()
        netGL.eval()
        netGH.eval()
        
        data_iter = iter(self.data_loader)
        step = 0
        while step < text_batch_num:
            step += 1
            ######################################################
            # (1) Prepare data and Compute text embeddings
            ######################################################
            data = data_iter.next()
            imgs, captions, cap_lens, class_ids, keys, Biclst, imgsblur, Bicblurlst = prepare_datablur(data,
                                                                                                       cfg=self.cfg)
            
            # for step, data in enumerate(self.data_loader, 0): # imgs, caps, cap_len, cls_id, key
            LRim = imgs[0]
            LRimb = imgsblur[0]
            
            # print('keys:', keys)
            s_tmp1 = '%s/%s/' % (s_tmp, savefile)
            save_dir = folder = s_tmp1[:s_tmp1.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            save_diratt = save_dir + '/att/'
            os.makedirs(save_diratt, exist_ok=True)
            
            if len(captions.shape) <= 1:
                captions = torch.unsqueeze(captions, 0)  # test batch = 1: 扩展维度 caption [18] to [1, 18]
            batch_size = captions.shape[0]
            print('caption length:', captions.shape[1], batch_size)
            
            for i in range(1):
                #######################################################
                # (1) Extract text embeddings
                ######################################################
                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                mask = (captions == 0)
                
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                
                #######################################################
                # (2) Generate fake images
                ######################################################
                fake_imgL, attention_maps, mu, logvar = netGL(LRim, sent_emb, words_embs, mask)  # SRb = imgsblur
                fine_im, a, one = netGH(LRim, fake_imgL, LRimb)
                
                for j in range(batch_size):
                    save_name = '%s/%s' % (save_dir, keys[0][:])  # 9
                    save_namea = '%s/%s' % (save_diratt, keys[0][:])  # 9
                    print('Save to: ', save_dir, save_name)
                    
                    im = np.round(
                        np.maximum(0, np.minimum(255, (fine_im[-1][0].data.cpu().numpy() + 1.0) * 127.5))).astype(
                        np.uint8)
                    Image.fromarray(np.transpose(im, (1, 2, 0))).save('%s_SR.png' % (save_name))  #
                    
                    # ## attenmap
                    im = fine_im[-1].detach().cpu()
                    cap_lens_np = cap_lens.cpu().data.numpy()
                    img_set, sentences = \
                        build_super_imagesall(im[j].unsqueeze(0), captions[j].unsqueeze(0),
                                              [cap_lens_np[j]], self.ixtoword, [attention_maps[0][j]],
                                              attention_maps[0].size(2))
                    Image.fromarray(img_set).save('%s.png' % (save_namea))


def rgb2y(rgb):
    h, w, d = rgb.shape
    rgb = np.float32(rgb) / 255.0
    y = rgb * (np.reshape([65.481, 128.553, 24.966], [1, 1, 3]) / 255.0)
    y = y[:, :, 0] + y[:, :, 1] + y[:, :, 2]
    y = np.reshape(y, [h, w]) + 16 / 255.0
    return np.uint8(y * 255 + 0.5)


def psnr(im1, im2):
    diff = np.float64(im1[:]) - np.float64(im2[:])
    rmse = np.sqrt(np.mean(diff ** 2))
    psnr = 20 * np.log10(255 / rmse)
    return psnr, rmse
