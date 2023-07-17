# coding: utf-8
import os
import errno
import numpy as np
from torch.nn import init

import torch
import torch.nn as nn

from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import skimage.transform

from miscc.config import cfg


# For visualization ################################################
COLOR_DIC = {0:[128,64,128],  1:[244, 35,232],
             2:[70, 70, 70],  3:[102,102,156],
             4:[190,153,153], 5:[153,153,153],
             6:[250,170, 30], 7:[220, 220, 0],
             8:[107,142, 35], 9:[152,251,152],
             10:[70,130,180], 11:[220,20, 60],
             12:[255, 0, 0],  13:[0, 0, 142],
             14:[119,11, 32], 15:[0, 60,100],
             16:[0, 80, 100], 17:[0, 0, 230],
             18:[0,  0, 70],  19:[0, 0,  0]}
FONT_MAX = 50


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    # fnt = None  # ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    fnt = ImageFont.truetype('C:/Users/Mloong/AppData/Local/Microsoft/Windows/Fonts/FreeMono.ttf', 50)# ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50) #
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list
def drawCaption_no_order(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    # get a font
    fnt = ImageFont.truetype('C:/Users/Mloong/AppData/Local/Microsoft/Windows/Fonts/FreeMono.ttf', 50)# ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50) #
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%s' % (word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_super_images(real_imgs, captions, ixtoword, attn_maps, att_sze, lr_imgs=None,
         batch_size=cfg.TRAIN.BATCH_SIZE, max_word_num=cfg.TEXT.WORDS_NUM):
    nvis = min(8, len(attn_maps))
    # nvis = 8
    real_imgs = real_imgs[:nvis]
    if lr_imgs is not None:
        lr_imgs = lr_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16  #
    else:
        vis_size = real_imgs.size(2)
        vis_size = 256  ## !!!!!!! ================= 让LR也能看清楚attention的文字 ===============!!!!!!!!!

    text_convas = \
        np.ones([batch_size * FONT_MAX,(max_word_num + 2) * (vis_size + 2), 3],dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i]

    real_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])
    if lr_imgs is not None:
        lr_imgs = nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(lr_imgs)
        # [-1, 1] --> [0, 1]
        lr_imgs.add_(1).div_(2).mul_(255)
        lr_imgs = lr_imgs.data.numpy()
        lr_imgs = np.round(np.maximum(0, np.minimum(255, lr_imgs)))
        # b x c x h x w --> b x h x w x c
        lr_imgs = np.transpose(lr_imgs, (0, 2, 3, 1))

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)  # [1,6,17,17]
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)  # 2 [1,1,17,17]
        attn = torch.cat([attn_max[0], attn], 1)  # [1,7,17,17]

        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))  # [7,17,17,3]
        num_attn = attn.shape[0]

        img = real_imgs[i]
        img = np.round(np.maximum(0, np.minimum(255, img)))
        if lr_imgs is None:
            lrI = img
        else:
            lrI = lr_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            # print('0 one_map.shape, vis_size, att_sze', one_map.shape, vis_size, att_sze)
            if (vis_size // att_sze) > 1:
                ## Server:
                one_map = skimage.transform.resize(one_map, (vis_size, vis_size), anti_aliasing=True, anti_aliasing_sigma=20)  # Upsample image.
                ## desktop:
                # one_map = skimage.transform.pyramid_expand(one_map, sigma=20,
                #         upscale=vis_size // att_sze)  # Upsample and then smooth image.
            # print('1 one_map.shape, vis_size, att_sze', one_map.shape, vis_size, att_sze)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                one_map *= 255

                PIL_im = Image.fromarray(np.uint8(img))
                # one_map = np.concatenate((np.expand_dims(one_map, 0),np.expand_dims(one_map, 0),np.expand_dims(one_map, 0)), 0)
                # print('2 one_map.shape', one_map.shape, img.shape)
                PIL_att = Image.fromarray(np.uint8(one_map))

                merged = Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)  # PIL_att, mask [256, 256]
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)

            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None

# 返回5张attention后的image连在一起的图像，上方有文字说明；返回当前图像的caption list
def build_super_images2(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, att_sze1=None, vis_size=256, topK=5):
    batch_size = real_imgs.size(0) # [1,3,256,256]
    if att_sze1 ==None:
        att_sze1 = att_sze
    max_word_num = np.max(cap_lens) # 19
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8) # [50, 4902=19*(256+2), 3]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    real_imgs = np.round(np.maximum(0, np.minimum(255, real_imgs)))
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3]) # [256,2,3]

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps) # [1, 22, 128, 128]

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size, off1=0) # [50, 4902=19*(256+2), 3]; [1,22]; [5450]; 256
    text_map = np.asarray(text_map).astype(np.uint8) # Image: [4902, 50]

    bUpdate = 1

    # 第几幅图片
    for i in range(num):
        try:
            attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze1) # [1,22,128,128]
        except:
            print('wrong!! shape attn_map', len(attn_maps), attn_maps[i].shape, 'att_size', att_sze)
            attn = attn_maps[i].cpu()
            attn = attn.view(-1, attn_maps[i].shape[0], attn_maps[i].shape[1], attn_maps[i].shape[2])
            attn = nn.Upsample(size=(att_sze, att_sze1), mode='bilinear')(attn)

            # attn = \
            #     nn.Upsample(size=(att_sze, att_sze1), mode='bilinear')(attn_maps[i])

            # attn = attn[i]

        attn = attn.view(-1, 1, att_sze, att_sze1)
        attn = attn.repeat(1, 3, 1, 1).data.numpy() # [22,3,128,128]
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i] # [256, 256, 3]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []

        # [22, 128, 128, 3] 对每一个attention image放大，01拉伸
        for j in range(num_attn):
            one_map = attn[j]  #
            mask0 = one_map > (2. * thresh) # [128, 128, 3]
            conf_score.append(np.sum(one_map * mask0)) # [19,]
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze) # 使用Upsample然后平滑图像
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map) # [19,256,256,3]
        sorted_indices = np.argsort(conf_score)[::-1] # [19]

        # 将每一个attention image与RGB图像融合显示，生成彩色Image
        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255  # [256, 256, 3]
            _, w, _ = one_map.shape
            one_map = np.resize(one_map, [256, 256, 3])
            #
            PIL_im = Image.fromarray(np.uint8(img)).resize((256, 256))  # [256, 256]
            PIL_att = Image.fromarray(np.uint8(one_map)).resize((256, 256))  # [256, 256]
            merged = Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0)) # [256, 256]
            mask = Image.new('L', (vis_size, vis_size), (180))  # [256, 256]
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3] # [256,256,3]

            row.append(np.concatenate([one_map, middle_pad], 1)) # 给原图像左右两列加一个黑边  [256,2,3]
            #
            row_merge.append(np.concatenate([merged, middle_pad], 1)) # merged:[256, 256, 3]
            # text_map [50, 4902, 3]
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :] # [50,258,3]
            row_txt.append(txt)
        # reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        # 存起来原始的19个图像，根据排序列表重新排序attention map
        for j in range(num_attn):
            idx = sorted_indices[j]
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        # 只取前5(topK)个
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None
# 返回所有attention后的image连在一起的图像，上方有文字说明；返回当前图像的caption list
def build_super_imagesall(real_imgs, captions, cap_lens, ixtoword,
                        attn_maps, att_sze, att_sze1=None, vis_size=256):
    batch_size = real_imgs.size(0) # [1,3,256,256]
    if att_sze1 ==None:
        att_sze1 = att_sze
    max_word_num = np.max(cap_lens) # 19
    text_convas = np.ones([batch_size * FONT_MAX,
                           max_word_num * (vis_size + 2), 3],
                           dtype=np.uint8) # [50, 4902=19*(256+2), 3]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear')(real_imgs)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    real_imgs = np.round(np.maximum(0, np.minimum(255, real_imgs)))
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3]) # [256,2,3]

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    img_set = []
    num = len(attn_maps) # [1, 22, 128, 128]

    # [50, 4902=19*(256+2), 3]; [1,22]; [5450]; 256
    text_map, sentences = drawCaption_no_order(text_convas, captions, ixtoword, vis_size, off1=0)
    text_map = np.asarray(text_map).astype(np.uint8) # Image: [4902, 50]

    bUpdate = 1

    # 第几幅图片
    for i in range(num):
        try:
            attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze1) # [1,22,128,128]
        except:
            print('wrong!! shape attn_map', len(attn_maps), attn_maps[i].shape, 'att_size', att_sze)
            attn = attn_maps[i].cpu()
            attn = attn.view(-1, attn_maps[i].shape[0], attn_maps[i].shape[1], attn_maps[i].shape[2])
            attn = nn.Upsample(size=(att_sze, att_sze1), mode='bilinear')(attn)

            # attn = \
            #     nn.Upsample(size=(att_sze, att_sze1), mode='bilinear')(attn_maps[i])

            # attn = attn[i]

        attn = attn.view(-1, 1, att_sze, att_sze1)
        attn = attn.repeat(1, 3, 1, 1).data.numpy() # [22,3,128,128]
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        num_attn = cap_lens[i]
        thresh = 2./float(num_attn)
        #
        img = real_imgs[i] # [256, 256, 3]
        row = []
        row_merge = []
        row_txt = []
        row_beforeNorm = []
        conf_score = []

        # [22, 128, 128, 3] 对每一个attention image放大，01拉伸
        for j in range(num_attn):
            one_map = attn[j]  #
            mask0 = one_map > (2. * thresh) # [128, 128, 3]
            conf_score.append(np.sum(one_map * mask0)) # [19,]
            mask = one_map > thresh
            one_map = one_map * mask
            if (vis_size // att_sze) > 1:
                one_map = skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze) # 使用Upsample然后平滑图像
            minV = one_map.min()
            maxV = one_map.max()
            one_map = (one_map - minV) / (maxV - minV)
            row_beforeNorm.append(one_map) # [19,256,256,3]

        # 将每一个attention image与RGB图像融合显示，生成彩色Image
        for j in range(num_attn):
            one_map = row_beforeNorm[j]
            one_map *= 255  # [256, 256, 3]
            _, w, _ = one_map.shape
            one_map = np.resize(one_map, [256, 256, 3])
            #
            PIL_im = Image.fromarray(np.uint8(img)).resize((256, 256))  # [256, 256]
            PIL_att = Image.fromarray(np.uint8(one_map)).resize((256, 256))  # [256, 256]
            merged = Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0)) # [256, 256]
            mask = Image.new('L', (vis_size, vis_size), (180))  # [256, 256]
            merged.paste(PIL_im, (0, 0))
            merged.paste(PIL_att, (0, 0), mask)
            merged = np.array(merged)[:, :, :3] # [256,256,3]

            row.append(np.concatenate([one_map, middle_pad], 1)) # 给原图像左右两列加一个黑边  [256,2,3]
            row_merge.append(np.concatenate([merged, middle_pad], 1)) # merged:[256, 256, 3]
            # text_map [50, 4902, 3]
            txt = text_map[i * FONT_MAX:(i + 1) * FONT_MAX,
                           j * (vis_size + 2):(j + 1) * (vis_size + 2), :] # [50,258,3]
            row_txt.append(txt)
        # 不reorder
        row_new = []
        row_merge_new = []
        txt_new = []
        # 存起来原始的19个图像，根据排序列表重新排序attention map
        for j in range(num_attn):
            idx = j
            row_new.append(row[idx])
            row_merge_new.append(row_merge[idx])
            txt_new.append(row_txt[idx])
        # 取前所有(topK)个
        topK = len(row_new)
        row = np.concatenate(row_new[:topK], 1)
        row_merge = np.concatenate(row_merge_new[:topK], 1)
        txt = np.concatenate(txt_new[:topK], 1)
        if txt.shape[1] != row.shape[1]:
            print('Warnings: txt', txt.shape, 'row', row.shape,
                  'row_merge_new', row_merge_new.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None

####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


###########  Visualization  ##################
def vis_layeror(layer, vgg16_conv, vgg16_deconv):
    """
    visualing the layer deconv result
    """
    num_feat = vgg16_conv.feature_maps[layer].shape[1]  # OrderedDict

    # set other feature map activations to zero
    new_feat_map = vgg16_conv.feature_maps[layer].clone()

    # choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]
        activation = torch.max(choose_map)
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)

    choose_map = new_feat_map[0, mark, :, :]
    max_activation = torch.max(choose_map)

    # make zeros for other feature maps
    if mark == 0:
        new_feat_map[:, 1:, :, :] = 0
    else:
        new_feat_map[:, :mark, :, :] = 0
        if mark != vgg16_conv.feature_maps[layer].shape[1] - 1:
            new_feat_map[:, mark + 1:, :, :] = 0

    choose_map = torch.where(choose_map == max_activation,
                             choose_map,
                             torch.zeros(choose_map.shape))

    # make zeros for ther activations
    new_feat_map[0, mark, :, :] = choose_map

    # print(torch.max(new_feat_map[0, mark, :, :]))
    print(max_activation)

    deconv_output = vgg16_deconv(new_feat_map, layer, mark, vgg16_conv.pool_locs)

    new_img = deconv_output.data.numpy()[0].transpose(1, 2, 0)  # (H, W, C)
    # normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)
    # cv2.imshow('reconstruction img ' + str(layer), new_img)
    # cv2.waitKey()
    return new_img, int(max_activation)

def vis_layer(layer, vgg16_conv, vgg16_deconv):
    all_feature = False  # 对SR的feature map全部区域做逆向去卷积
    original_feature = True  # False  # 原始的，只对有激活值的特征层的部分区域做去卷积

    meanfeature = False  # True  # 对有最大激活值的层的大于该层均值的区域做去卷积
    fourfeature = True  # 对有最大激活值的层 大于该层四分位数值的区域做去卷积

    #### !! 输出的new_img已经是0-255之间的uint8格式！！
    num_feat = vgg16_conv.feature_maps[layer].shape[1]  # 64

    #### set other feature map activations to zero
    new_feat_map = vgg16_conv.feature_maps[layer].clone()  # [1, 64, 32, 32]

    #### choose the max activations map
    act_lst = []
    for i in range(0, num_feat):
        choose_map = new_feat_map[0, i, :, :]  # [32, 32]
        activation = torch.max(choose_map)  # 0.4378
        act_lst.append(activation.item())

    act_lst = np.array(act_lst)
    mark = np.argmax(act_lst)  # 62

    choose_map = new_feat_map[0, mark, :, :]  # 从所有feature map里面选择响应度最高的feature map
    max_activation = torch.max(choose_map)  # 0.6336
    mean_act = torch.mean(choose_map)  # 0.2003
    four_act = np.percentile(choose_map.detach().cpu().numpy(), 10, interpolation='midpoint')  # 0.16315

    if all_feature:  # ## full feature ##
        new_feat_map = new_feat_map
    elif original_feature:
        #### make zeros for other feature maps # 除了响应度最高的feature map外，其它feature map设为0
        if mark == 0:
            new_feat_map[:, 1:, :, :] = 0
        else:
            new_feat_map[:, :mark, :, :] = 0
            if mark != vgg16_conv.feature_maps[layer].shape[1] - 1:
                new_feat_map[:, mark + 1:, :, :] = 0
        if meanfeature:
            choose_map = torch.where(choose_map >= mean_act, choose_map, torch.zeros(choose_map.shape).cuda())
        elif fourfeature:
            choose_map = torch.where(choose_map >= four_act, choose_map, torch.zeros(choose_map.shape).cuda())
        else: #### torch.where(cond,x,y) 给定一个条件cond，满足条件的取x对应位置元素，不满足的取y对应元素
            choose_map = torch.where(choose_map == max_activation, choose_map, torch.zeros(choose_map.shape).cuda())
        # print(choose_map.detach().cpu().numpy(), torch.max(choose_map).detach().cpu().numpy()) #  [...[0. 0. 0. ... 0. 0. 0.]] 0.63356006

        #### make zeros for ther activations
        new_feat_map[0, mark, :, :] = choose_map

    deconv_output = vgg16_deconv(new_feat_map, layer)
    new_img = deconv_output.data.cpu().numpy()[0].transpose(1, 2, 0)  # (H, W, C)  #
    #### normalize
    new_img = (new_img - new_img.min()) / (new_img.max() - new_img.min()) * 255
    new_img = new_img.astype(np.uint8)  # [32, 32, 3]
    return new_img, int(max_activation)
