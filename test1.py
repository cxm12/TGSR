# coding: utf-8
from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from datasets import *
from trainer_objective import condGANTrainer as trainerob

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)
data = 'face'
no_StopWord = False
OnlyColor = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train a AttnGAN network')
    if data == 'coco':
        parser.add_argument('--cfg', dest='cfg_file', default='cfg/eval_cocoSR_attn2.yml', type=str)
    elif data == 'bird':
        parser.add_argument('--cfg', dest='cfg_file', default='cfg/eval_birdSR_attn2.yml', type=str)
    elif data == 'face':
        parser.add_argument('--cfg', dest='cfg_file', default='cfg/eval_faceSR_attn2.yml', type=str)
    elif data == 'flower':
        parser.add_argument('--cfg', dest='cfg_file', default='cfg/eval_flowerSR_attn2.yml', type=str)
    elif data == 'urban100':
        parser.add_argument('--cfg', dest='cfg_file', default='cfg/eval_div2kSR_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def gen_examplePickle(wordtoix, ixtoword, namein='example.txt', namecp='example.pickle'):
    from nltk.corpus import stopwords, color
    stop_words = set(stopwords.words("english"))
    color_words = set(color.words("english"))
    color_words.add('bird')
    color_words.add('birds')  #
    
    filepath = '%s/%s' % (cfg.DATA_DIR, namein)
    
    if not os.path.exists(cfg.DATA_DIR + '/' + namecp):
        print('creating~', cfg.DATA_DIR + namecp, 'for image name', namein)
        with open(filepath, "r") as f:
            filenames = f.read().encode('utf8').decode('utf8').split('\n')
            captions_new = []
            all_captions = []
            for name in filenames:
                if len(name) == 0:
                    continue
                
                print('caption of ', name)
                cap_path = '%s/%s.txt' % (cfg.DATA_DIR, name)
                with open(cap_path, "r") as f:
                    captions = f.read().encode('utf8').decode('utf8').split('\n')
                    cnt = 0
                    for cap in captions:
                        if len(cap) == 0:
                            continue
                        
                        cap = cap.replace("\ufffd\ufffd", " ")
                        # picks out sequences of alphanumeric characters as tokens and drops everything else
                        tokenizer = RegexpTokenizer(r'\w+')
                        tokens = tokenizer.tokenize(cap.lower())
                        # print('tokens', tokens)
                        if len(tokens) == 0:
                            print('cap', cap)
                            continue
                        
                        if no_StopWord:
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
                        if OnlyColor:
                            tokens = [w for w in tokens if w in color_words]
                            if len(tokens) == 0:
                                print('cap', cap)
                                cap = 'bird'
                                cap = cap.replace("\ufffd\ufffd", " ")  # 一句英文话
                                tokenizer = RegexpTokenizer(r'\w+')
                                tokens = tokenizer.tokenize(cap.lower())
                                tokens = [w for w in tokens if w in color_words]
                        tokens_new = []
                        for t in tokens:
                            t = t.encode('ascii', 'ignore').decode('ascii')
                            if len(t) > 0:
                                tokens_new.append(t)
                        all_captions.append(tokens_new)
                        cnt += 1
                        if cnt == 10:
                            break
            cap_array = all_captions  # sentence
            
            for t in cap_array:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                
                captions_new.append(rev)  #
            
            filepath = os.path.join(cfg.DATA_DIR, namecp)
            import pickle
            if not os.path.isfile(filepath):
                with open(filepath, 'wb') as f:
                    pickle.dump([captions_new, ixtoword, wordtoix], f, protocol=2)
                    print('Save to: ', filepath)
    else:
        print('Existing~', cfg.DATA_DIR + namecp, 'for image name', namein)
    
    return namecp


def test_example():
    if data == 'face':
        datasettrain = TextSRDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE,
                                     transform=image_transform, cfg=cfg)
        cap = gen_examplePickle(datasettrain.wordtoix, datasettrain.ixtoword, 'testset.txt')
        dataset = TextexampleSRDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE, cfg=cfg,
                                       data='face', cap=cap, txt='testset.txt')
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, drop_last=True, shuffle=True,
                                             num_workers=int(cfg.WORKERS))
    
    start_t = time.time()
    print('datasetSRtest.n_words', dataset.n_words)
    algo = trainerob(output_dir, dataloader, dataset.n_words, dataset.ixtoword, cfg)
    algo.gen_exampleSRHL(savefile='testset')
    
    end_t = time.time()
    print('Total time: ', end_t - start_t)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False
    
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    
    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)
    
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, cfg.METHOD)
    
    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize), transforms.RandomHorizontalFlip()])
    
    test_example()
