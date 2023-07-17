# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
# from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
import glob
import cv2 as cv
from PIL import ImageFilter
import math
import pickle

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data, cfg):
    imgs, captions, captions_lens, class_ids, keys, bic = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    real_imgsbic = []
    for i in range(len(imgs)):
        bic[i] = bic[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgsbic.append(Variable(bic[i]).cuda())
        else:
            real_imgsbic.append(Variable(bic[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys, real_imgsbic]


def prepare_datablur(data, cfg):
    imgs, captions, captions_lens, class_ids, keys, bic, blur, bicblur = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_imgs = []
    real_blur = []
    real_imgsbic = []
    real_imgsbicblur = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        blur[i] = blur[i][sorted_cap_indices]
        bic[i] = bic[i][sorted_cap_indices]
        bicblur[i] = bicblur[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
            real_blur.append(Variable(blur[i]).cuda())
            real_imgsbic.append(Variable(bic[i]).cuda())
            real_imgsbicblur.append(Variable(bicblur[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))
            real_blur.append(Variable(blur[i]))
            real_imgsbic.append(Variable(bic[i]))
            real_imgsbicblur.append(Variable(bicblur[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys, real_imgsbic, real_blur, real_imgsbicblur]


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None, cfg=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    
    if transform is not None:
        img = transform(img)
    
    ret = []
    bic = []
    lrimg = transforms.Resize(imsize[0])(img)
    
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                # re_img = transforms.Resize(imsize[i, 0])(img)
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))
            
            if i < (cfg.TREE.BRANCH_NUM):
                bicim = transforms.Resize(imsize[i])(lrimg)
            bic.append(normalize(bicim))
    
    return ret, bic
   
    
def get_imgs_blur(img_path, imsize, bbox=None, transform=None, normalize=None, cfg=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    retb = []
    bic = []
    bicb = []
    lrimg = transforms.Resize(imsize[0])(img)

    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img) # re_img = transforms.Resize(imsize[i, 0])(img) #
            else:
                re_img = img

            # GBlur = MyGaussianBlur(radius=15, sigema=2.0)  # 声明高斯模糊类
            # temp = GBlur.template()  # 得到滤波模版
            # re_imgblur= GBlur.filter(re_img, temp)
            re_imgblur= re_img.filter(ImageFilter.GaussianBlur(radius=2))  # cv.GaussianBlur(re_img, 15, 2.0) #

            ret.append(normalize(re_img))
            retb.append(normalize(re_imgblur))

            if i < (cfg.TREE.BRANCH_NUM):
                bicim = transforms.Resize(imsize[i])(lrimg)
                bicblur = bicim.filter(ImageFilter.GaussianBlur(radius=2))
            bicb.append(normalize(bicblur))
            bic.append(normalize(bicim))

    return ret, bic, retb, bicb


def get_imgsexampletest(img_path, scale=4, transform=None, normalize=None, cfg=None):
    img = Image.open(img_path).convert('RGB')
    
    if transform is not None:
        img = transform(img)

    ret = []
    bic = []
    imsize = np.array(img.size)
    h = imsize[1]
    w = imsize[0]
    h = h // scale*scale
    w = w // scale*scale
    img = img.crop([0, 0, w, h])
    imsize0 = [h // scale, w // scale]

    lrimg = transforms.Resize(imsize0)(img)
    bicsize = imsize0
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):  # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize0)(img)
                imsize0 = [imsize0[0] * 2, imsize0[1] * 2]
            else:
                re_img = img
            ret.append(normalize(re_img))
            if i < (cfg.TREE.BRANCH_NUM):
                bicim = transforms.Resize(bicsize)(lrimg)
                bicsize = [bicsize[0] * 2, bicsize[1] * 2]
            bic.append(normalize(bicim))

    return ret, bic


def get_imgsexampletestblur(img_path, scale=4, transform=None, normalize=None, cfg=None):
    img = Image.open(img_path).convert('RGB')

    if transform is not None:
        img = transform(img)

    ret = []
    retb = []
    bic = []
    bicb = []
    imsize = np.array(img.size)
    h = imsize[1]
    w = imsize[0]
    h = h//scale*scale
    w = w//scale*scale
    img = img.crop([0, 0, w, h])
    # imsize0 = np.array([h // scale, w // scale])
    imsize0 = [h // scale, w // scale]

    lrimg = transforms.Resize(imsize0)(img)  # print('img.size, lrimg.size', img.size, lrimg.size)
    bicsize = imsize0
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):  # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize0)(img)
                imsize0 = [imsize0[0] * 2, imsize0[1] * 2]
            else:
                re_img = img

            re_imgblur= re_img.filter(ImageFilter.GaussianBlur(radius=2))
            retb.append(normalize(re_imgblur))

            ret.append(normalize(re_img))
            if i < (cfg.TREE.BRANCH_NUM):
                bicim = transforms.Resize(bicsize)(lrimg)
                bicsize = [bicsize[0] * 2, bicsize[1] * 2]
                bicblur = bicim.filter(ImageFilter.GaussianBlur(radius=2))
            bicb.append(normalize(bicblur))
            bic.append(normalize(bicim))

    return ret, bic, retb, bicb


## ===================== image to caption ===================== ##
class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64,
                 transform=None, target_transform=None,cfg=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train') # 'images/train2017')
        test_names = self.load_filenames(data_dir, 'test') # 'images/train2017') #
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs, bicimgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key, bicimgs

    def __len__(self):
        return len(self.filenames)


## ============================== image SR ========================= ##
class TextSRDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64,
                 transform=None, target_transform=None, cfg=None,
                 nostop=False, onlycolor=False):
        self.onlycolor = onlycolor
        self.nostop = nostop
        self.transform = transform
        self.split = split
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.cfg = cfg

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size*2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        # caption: list(88550){list(11-30){3066,4217,2622,3066,...,4839}}
        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        # _ =  self.load_text_data_exampletest(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path, delim_whitespace=True, header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
    
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions
    
    def load_captions_stopwords(self, data_dir, filenames):
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))  # ['more', 'him', 'being', ..., ]
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    ## -------------- 停用词 --------- ###
                    tokens = [w for w in tokens if not w in stop_words]
                    if len(tokens) == 0:
                        print('cap', cap)
                        cap = captions[0]
                        cap = cap.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cap.lower())
                        tokens = [w for w in tokens if not w in stop_words]
                        if len(tokens) == 0:
                            print('still error : cap', cap)
                            continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')  # 'a'
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))
        return all_captions
    
    def load_captions_c(self, data_dir, filenames):
        from nltk.corpus import color
        color_words = set(color.words("english"))  # 203 ['auburn', 'caramel', ..., ]
        color_words.add('bird')
        color_words.add('birds') #
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)

                    tokens = [w for w in tokens if w in color_words]
                    if len(tokens) == 0:
                        print('cap', cap)
                        cap = 'bird'
                        cap = cap.replace("\ufffd\ufffd", " ")
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cap.lower())
                        tokens = [w for w in tokens if w in color_words]
                        if len(tokens) == 0:
                            print('still error : no color words in text cap', cap)
                            continue
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')  # 'a'
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))
        return all_captions
    
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions5450.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        
        if not os.path.isfile(filepath):
            print('No caption.pickle! Create now!')
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions) # 88550 list 存单词对应的dictionary的数值表示

            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath, 'number of words', n_words)
        if split == 'train':
            # a list of list: each list contains  # the indices of words in a sentence
            captions = train_captions  # list(88550){list(11-30){3066,4217,2622,3066,...,4839}}
            filenames = train_names  # list(8855){'002.Laysan_Albatross/Laysan_Albatross_0002_1027'...}
        else:  # split=='test'
            captions = test_captions  # list[list]
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words
    
    # buid pickle of example test file
    def build_dictionary_example(self, train_captions, test_captions, examp):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions + examp
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        ex_captions_new = []
        for t in examp:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            ex_captions_new.append(rev)

        return [train_captions_new, test_captions_new, ex_captions_new,
                ixtoword, wordtoix, len(ixtoword)]
    def load_text_data_exampletest(self, data_dir, split):
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        filepath = '%s/example_filenames1.txt' % (data_dir)
        text = []
        with open(filepath, "r") as f:
            filenames = f.read().encode('utf8').decode('utf8').split('\n')
            for name in filenames:
                if len(name) == 0:
                    continue
                filepath = '%s/%s.txt' % (data_dir, name)
                text.append(filepath)
        ex_names = []
        for i in range(len(text)):
            nm = text[i]
            nm1 = nm[len('../data/birds/text/'):-4]
            ex_names.append(nm1)

        filepath = os.path.join(data_dir, 'exampletest_captions_notraintest.pickle')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            example_captions = self.load_captions(data_dir, ex_names)

            # n_words = 5598
            train_captions, test_captions,example_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary_example(train_captions, test_captions,example_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([example_captions,
                             ixtoword, wordtoix], f, protocol=2)
                # pickle.dump([train_captions, test_captions, example_captions,
                #              ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)

        return filenames
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    
    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames
    
    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]

        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        img_name = '%s/images/%s.jpg' % (data_dir, key)

        # random select a sentence
        if self.split == 'train':
            sent_ix = random.randint(0, self.embeddings_num)
        else:
            sent_ix = 0
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        imgs, bicimgs, imgsblur, bicblurim = get_imgs_blur(img_name, self.imsize, bbox, self.transform, normalize=self.norm, cfg=self.cfg)
        return imgs, caps, cap_len, cls_id, key, bicimgs, imgsblur, bicblurim

    def __len__(self):
        return len(self.filenames)


class TextfaceDataset(data.Dataset):
    def __init__(self, data_dir, data_dirim, split='train', base_size=64,
                 transform=None, target_transform=None, cfg=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.data_dirim = data_dirim
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
    
    def load_captions(self, data_dir, train=True):
        ## load attributions
        all_captions = []
        namelst = []
        file = open(data_dir+'list_attr_celeba.txt')  # f = open('caps1.txt', 'w')
        for line in file:
            attr_list = line.split(' ')  # 41
            attr_list = attr_list[:-1]  # 40
            break
        for line in file:
            attr = line.split(' ')  # 41
            tokens_new = []
            if train:
                if attr[11] == '0':
                    namelst.append(attr[0][:-3]+'png')
                    for i in range(1, len(attr)):
                        if attr[i] == '1' or attr[i] == '1\n':
                            tokens_new.append(attr_list[i - 1])
                    all_captions.append(tokens_new)
            else:
                if attr[11] == '1':
                    namelst.append(attr[0][:-3]+'png')
                    for i in range(1, len(attr)):
                        if attr[i] == '1' or attr[i] == '1\n':
                            tokens_new.append(attr_list[i - 1])
                    all_captions.append(tokens_new)

        return all_captions, namelst

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]
    
    def load_caps(self, data_dir, train=True):
        ## load sentences
        all_captions = []
        namelst = []
        file = open(data_dir + 'caps.txt')
        i = 0
        for linestence in file:
            sentenceall = linestence[linestence.find('\t') + 1:-1].split('|')

            namelst.append(linestence[:7] + 'png')
            for i in range(5):  # len(sentenceall)
                if len(sentenceall) > i:
                    sentenceall[i] = sentenceall[i][:-1].replace(',', '').lower()
                    words = sentenceall[i].split(' ')
                    all_captions.append(words)
                else:
                    sentenceall[i % len(sentenceall)] = sentenceall[i % len(sentenceall)][:]
                    words = sentenceall[i % len(sentenceall)].split(' ')
                    all_captions.append(words)
            i += 1
        if not train:
            all_captions = all_captions[:25]
            namelst = namelst[:5]

        return all_captions, namelst
    
    def build_dictionary5(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        if not os.path.isfile(filepath):
            print('strat creat', filepath)
            # train_captions, train_names = self.load_caps(data_dir+'/text/', train=True)  # 192287
            # test_captions, test_names = self.load_caps(data_dir+'/text/', train=False)  # 10312
            train_captions, train_names = self.load_captions(data_dir + '/text/', train=True)  # 192287
            test_captions, test_names = self.load_captions(data_dir + '/text/', train=False)  # 10312

            train_captions, test_captions, ixtoword, wordtoix, n_words = self.build_dictionary(train_captions, test_captions)
                # self.build_dictionary5(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, train_names, test_captions, test_names, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, train_names, test_captions, test_names = x[0], x[1], x[2], x[3]
                ixtoword, wordtoix = x[4], x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':  # a list of list: each list contains the indices of words in a sentence
            captions = train_captions[150:]  # 192287
            filenames = train_names[150:]
        else:
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    
    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        bbox = None

        # random select a sentence
        if self.split == 'train':
            sent_ix = random.randint(0, self.embeddings_num) # face:self.embeddings_num=1
        else:
            sent_ix = 0
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        img_name = '%s/Img/%s' % (self.data_dirim, key)
        imgs, bicimgs, imgsblur, bicblurim = get_imgs_blur(img_name, self.imsize, bbox, self.transform, normalize=self.norm, cfg=self.cfg)
        return imgs, caps, cap_len, cls_id, key, bicimgs, imgsblur, bicblurim

    def __len__(self):
        return len(self.filenames)


class TextflowerDataset(data.Dataset):
    def __init__(self, data_dir, data_dirim, split='train', base_size=64, transform=None, target_transform=None, cfg=None):
        self.cfg = cfg
        self.scale = 8
        self.split = split
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.data_dirim = data_dirim
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        
    def load_captions(self, data_dir, train):
        all_captions = []
        namelst = []
        txtfile = glob.glob(data_dir+"/*.txt")
        if train:
            start = 0
            end = len(txtfile)-3
        else:
            start = len(txtfile)-3
            end = len(txtfile)
        for i in range(start, end):
            txtnm = txtfile[i]
            name = txtnm[len(data_dir):-4]+'.jpg'
            namelst.append(name)
            with open(txtnm, "r") as f:
                captions = f.read().encode('utf8').decode('utf8').replace('.', '').replace(',', '').split('\n')
                captions = captions[:10]
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    tokens_new = []

                    if cap == '{}':
                        all_captions.append(all_captions[-1])
                        cnt += 1
                        continue

                    if len(tokens) == 0:
                        print('cap', cap)
                        continue
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d' % (txtnm, cnt))
        if len(all_captions) != len(namelst)*10:
            print('Wrong !!! image DO NOT contain 10 captions', data_dir)
        return all_captions, namelst
    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]  # 5618

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new, ixtoword, wordtoix, len(ixtoword)]
    
    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions1.pickle')
        if not os.path.isfile(filepath):
            train_captions, train_names = [], []
            test_captions, test_names = [], []

            for i in range(102):
                print('load class_00%03d'%(i+1))
                train_captions1, train_names1 = self.load_captions(data_dir + '/text_c10/class_00%03d'%(i+1) +'/', train=True)
                test_captions1, test_names1 = self.load_captions(data_dir + '/text_c10/class_00%03d'%(i+1) +'/', train=False)
                train_captions+=train_captions1
                train_names+=train_names1
                test_captions += test_captions1
                test_names += test_names1

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, train_names, test_captions, test_names, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, train_names, test_captions, test_names = x[0], x[1], x[2], x[3]  # 7679/510
                ixtoword, wordtoix = x[4], x[5]
                del x
                n_words = len(ixtoword)
                print(ixtoword[5295])
                print('Load from: ', filepath)
        if split == 'train':  # a list of list: each list contains the indices of words in a sentence
            captions = train_captions  # 192287
            filenames = train_names
        else:  # split=='test'
            captions = test_captions[0:len(test_captions):self.embeddings_num*3]  # 10312   202599
            filenames = test_names[0:len(test_names):3]
        return filenames, captions, ixtoword, wordtoix, n_words
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    
    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        bbox = None
        
        img_name = '%s/%s' % (self.data_dirim, key)

        # random select a sentence
        if self.split == 'train':
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
        else:
            new_sent_ix = index
        caps, cap_len = self.get_caption(new_sent_ix)

        imgs, bicimgs, imgsblur, bicblurim = get_imgs_blur(img_name, self.imsize, bbox, self.transform, normalize=self.norm, cfg=self.cfg)
        return imgs, caps, cap_len, cls_id, key[8:], bicimgs, imgsblur, bicblurim
        
    def __len__(self):
        return len(self.filenames)


class TextcocoDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64, transform=None, target_transform=None, cfg=None):
        self.transform = transform
        self.scale = 4
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        self.bbox = None
        split_dir = os.path.join(data_dir, split)
        self.split = split
        self.cfg = cfg

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().decode('utf8').split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions#[:self.embeddings_num]  #
            filenames = test_names#[:1]
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f,encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        key = self.filenames[index]
        cls_id = self.class_id[index]
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        if self.split == 'train':
            img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
        else:
            img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)

        if self.split =='train':
            imgs, bicimgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm, cfg=self.cfg)
        else:
            imgs, bicimgs = get_imgsexampletest(img_name, self.scale, self.transform, normalize=self.norm, cfg=self.cfg)

        # random select a sentence
        if self.split == 'train':
            sent_ix = random.randint(0, self.embeddings_num)
        else:
            sent_ix = 0
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return imgs, caps, cap_len, cls_id, key, bicimgs

    def __len__(self):
        return len(self.filenames)


class TextexampleSRDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64,
                 cfg=None, data='',
                 cap='exampletest_captions_notraintest.pickle', txt='example_filenames1.txt'):
        self.dataset = data
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        # self.transform = None
        self.transform = transforms.Compose([
            transforms.Resize(int(imsize * 72 / 64)),
            transforms.CenterCrop(imsize)
            ])

        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.cfg = cfg

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        if self.dataset != 'face':
            filepath = '%s/%s' % (cfg.DATA_DIR, txt)
            text = []
            with open(filepath, "r") as f:
                filenames = f.read().encode('utf8').decode('utf8').split('\n')
                for name in filenames:
                    if len(name) == 0:
                        continue
                    filepath = '%s/%s.txt' % (cfg.DATA_DIR, name)
                    text.append(filepath)
            self.text = text
            self.filenames = text
        else:
            self.text = txt
            self.dataim = self.cfg.DATA_DIR + '/Img/changetxt_im1/'  # testset
        self.data_dir = data_dir
        self.bbox = None

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(self.text, cap)
        self.number_example = len(self.filenames)
        split_dir = os.path.join(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        
    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
    
    def load_text_data(self, text, cap='example_captions1.pickle'):
        filepath = os.path.join(self.data_dir, cap)  # exampletest_captions.pickle
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            examplecaptions = x[0]  ### 0 train  1 test ###
            ixtoword, wordtoix = x[1], x[2]  # x[3], x[4] #
            del x
            n_words = len(ixtoword)  # 5598
            print('Load from: ', filepath)

        captions = examplecaptions
        if self.dataset == 'flower':
            filenames = []
            for fl in text:
                filenames.append(fl.replace('text_c10', 'jpg1').replace('txt', 'jpg'))
        elif self.dataset == 'face':
            filenames = []
            file = open(self.cfg.DATA_DIR + '/text/changetxt_text/' + text)
            for line in file:
                break
            for line in file:
                attr = line.split(' ')
                filenames.append(self.dataim + attr[0][:-3] + 'png')
        else:
            filenames = text
        return filenames, captions, ixtoword, wordtoix, n_words
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    
    def get_caption(self, sent_ix):  # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        if self.dataset == 'face':
            key = self.filenames[index]
            cls_id = self.class_id[index]
            img_name = key

            # random select a sentence
            sent_ix = 0  # random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
            caps, cap_len = self.get_caption(new_sent_ix)

            imgs, bicims = get_imgsexampletest(img_name, self.imsize[-1] // self.imsize[0], self.transform, normalize=self.norm, cfg=self.cfg)
            return imgs, caps, cap_len, cls_id, key[len(self.dataim):], bicims
    
    def __len__(self):
        return len(self.filenames)
    

class MyGaussianBlur():
    def __init__(self, radius=1, sigema=1.5):
        self.radius=radius
        self.sigema=sigema
    def calc(self,x,y):
        res1=1/(2*math.pi*self.sigema*self.sigema)
        res2=math.exp(-(x*x+y*y)/(2*self.sigema*self.sigema))
        return res1*res2
    def template(self):
        sideLength=self.radius*2+1
        result = np.zeros((sideLength, sideLength))
        for i in range(sideLength):
            for j in range(sideLength):
                result[i,j]=self.calc(i-self.radius, j-self.radius)
        all=result.sum()
        return result/all
    def filter(self, image, template):
        arr=np.array(image)
        height=arr.shape[0]
        width=arr.shape[1]
        newData=np.zeros((height, width))
        for i in range(self.radius, height-self.radius):
            for j in range(self.radius, width-self.radius):
                t=arr[i-self.radius:i+self.radius+1, j-self.radius:j+self.radius+1]
                a= np.multiply(t, template)
                newData[i, j] = a.sum()
        newImage = Image.fromarray(newData)
        return newImage


class TextSRGTDataset(data.Dataset):
    def __init__(self, wordtoix, ixtoword, data_dir, cfg, target_transform=None,
                 datadir_im = '../data/birds/test/HR/', data='bird', txt='testset.txt'):
        imsize = 256
        # self.transform = None
        self.transform = transforms.Compose([ # transforms.Resize(int(imsize * 72 / 64)),
            transforms.CenterCrop(imsize)
            ])

        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.cfg = cfg

        self.data = data
        filepath = '%s/%s' % (cfg.DATA_DIR, txt)
        self.filenames = []
        self.text = []
        self.data_dir = data_dir
        self.dataim = datadir_im

        if self.data == 'face':
            file = open('%s/%s' % (cfg.DATA_DIR, txt))
            all_captions = []
            for line in file:
                attr_list = line.split(' ')  # 41
                attr_list = attr_list[:-1]  # 40
                break
            for line in file:
                attr = line.split('|')
                self.filenames.append(attr[0])
                attr = attr[1].split(' ')  # 41
                tokens_new = []
                for i in range(1, len(attr)):
                    if attr[i] == '1' or attr[i] == '1\n':
                        tokens_new.append(attr_list[i - 1])
                all_captions.append(tokens_new)
            self.text = all_captions  # sentence
            self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_dataface(wordtoix, ixtoword)
        else:
            with open(filepath, "r") as f:
                filenames = f.read().encode('utf8').decode('utf8').split('\n')
                for name in filenames:
                    if len(name) == 0:
                        continue
                    name1 = name.split('|')
                    self.filenames.append(name1[0])
                    self.text.append(name1[1:][0])
            self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(wordtoix, ixtoword)
        # self.filenames = self.get_filenames([datadir_im])
        self.class_id = self.load_class_id('test', len(self.filenames))
        self.number_example = len(self.filenames)
    def load_text_data(self, wordtoix, ixtoword, cap = 'testset_captions.pickle'):
        n_words = len(ixtoword)  # 5598
        filepath = os.path.join(self.data_dir, cap)
        if not os.path.exists(os.path.join(self.data_dir, cap)):
            print('creating~', os.path.join(self.data_dir, cap))
            captions_new = []
            all_captions = []
            cnt = 0
            for cap in self.text:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ").replace(',', '').replace('.', '')
                # picks out sequences of alphanumeric characters as tokens and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(cap.lower())
                if len(tokens) == 0:
                    print('cap', cap)
                    continue
                tokens_new = []
                for t in tokens:
                    t = t.encode('ascii', 'ignore').decode('ascii')
                    if len(t) > 0:
                        tokens_new.append(t)
                all_captions.append(tokens_new)
                cnt += 1
                # if cnt == 10:
                #     break
            cap_array = all_captions  # sentence

            for t in cap_array:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                captions_new.append(rev)

            with open(filepath, 'wb') as f:
                pickle.dump([captions_new, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
            return captions_new, ixtoword, wordtoix, n_words
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                captions = x[0]  ### 0 train  1 test ###
                ixtoword, wordtoix = x[1], x[2]
                del x
                # print('Load from: ', filepath)
                return captions, ixtoword, wordtoix, n_words
            
    def load_text_dataface(self, wordtoix, ixtoword, cap = 'testset_captions.pickle'):
        n_words = len(ixtoword)  # 5598
        filepath = os.path.join(self.data_dir, cap)
        if not os.path.isfile(filepath):
            print('strat creat', filepath)
            train_captions = self.text

            train_captions_new = []
            for t in train_captions:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                train_captions_new.append(rev)

            with open(filepath, 'wb') as f:
                pickle.dump([train_captions_new, ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
            return train_captions_new, ixtoword, wordtoix, n_words
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                captions = x[0]  ### 0 train  1 test ###
                ixtoword, wordtoix = x[1], x[2]
                del x
                # print('Load from: ', filepath)
                return captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    def __getitem__(self, index):
        cls_id = self.class_id[index]
        key = self.filenames[index]
        img_name = self.dataim + key + '_x4_SR.png'  # jpg_HR.  [:-3] + '_s-1.png'  # + 's_SR.png'  # + 's_HR.png'  # _x4_SR_x2_SR
        # print('img_name', img_name)

        ###  sentence
        # caps, cap_len = self.get_caption(new_sent_ix)
        sent_caption = np.asarray(self.captions[index]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM

        img = Image.open(img_name).convert('RGB')
        imgs = self.norm(self.transform(img))
        return [imgs], x, x_len, key, cls_id
    def __len__(self):
        return len(self.filenames)


class TextexampleSRDataset_meaningless(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64, cfg=None,
                 cap='exampletest_captions_notraintest.pickle'):
        self.dataset = data
        imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
        self.transform = transforms.Compose([  # transforms.Scale(int(imsize * 72 / 64)),
            transforms.Resize(int(imsize * 72 / 64)),  #
            transforms.CenterCrop(imsize)  # transforms.RandomCrop(imsize)
        ])
        
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.cfg = cfg
        
        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2
        
        self.data = []
        self.datapath = '../data/testset/faceHR/'
        self.filenames = glob.glob(self.datapath + '*.png')
        self.data_dir = data_dir
        
        self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(cap)
        self.number_example = len(self.filenames)
        split_dir = os.path.join(data_dir, split)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
    
    def load_text_data(self, cap='example_captions1.pickle'):
        # filepath = os.path.join('../data/birds', cap)  # exampletest_captions.pickle
        filepath = os.path.join('../data/face', cap)  # exampletest_captions.pickle
        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            captions = [[0], [0], [0]]  # x[0]  ### 0 train  1 test ###
            ixtoword, wordtoix = x[1], x[2]  # x[3], x[4] #
            del x
            n_words = len(ixtoword)  # 5598
            print('Load from: ', filepath)
        
        return captions, ixtoword, wordtoix, n_words
    
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='iso-8859-1')
        else:
            class_id = np.arange(total_num)
        return class_id
    
    def get_caption(self):  # a list of indices for a sentence
        # print('self.ixtoword[len(self.ixtoword)-1]', self.ixtoword[len(self.ixtoword)-1], self.ixtoword[3])
        sent_caption = np.asarray([len(self.ixtoword)-1, len(self.ixtoword)-1, len(self.ixtoword)-1, len(self.ixtoword)-1, len(self.ixtoword)-1]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.cfg.TEXT.WORDS_NUM
        return x, x_len
    
    def __getitem__(self, index):
        key = self.filenames[index][len(self.datapath):]
        cls_id = self.class_id[index]
        img_name = self.filenames[index]  # '%s/images/%s.jpg' % (self.data_dir, key[len('../data/coco/text/'):-4])
        print('img_name', img_name, 'key:', key)
    
        ### random select a sentence
        caps, cap_len = self.get_caption()
    
        imgs, bicims = get_imgsexampletest(img_name, self.imsize[-1] // self.imsize[0], self.transform,
                                               normalize=self.norm, cfg=self.cfg)
        return imgs, caps, cap_len, cls_id, key[key.rfind('/') + 1:].replace('.txt', ''), bicims

    def __len__(self):
        return len(self.filenames)
