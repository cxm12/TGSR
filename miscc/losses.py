import torch
import torch.nn as nn

import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention
import torch.nn.functional as F

server = 0
# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8) # [16]
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0) # [16]
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)  # [1, b, 256]
        rnn_code = rnn_code.unsqueeze(0)  # [1, b, 256]

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True) # [1, b, 1]
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))  # [1, b, b]
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))  # [1, b, b]
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3  # [1, b, b]

    # --> batch_size x batch_size
    scores0 = scores0.squeeze(dim=0)  # scores0.squeeze()  # [b, b]
    if class_ids is not None:
        if server:
            scores0.data.masked_fill_(masks.bool(), -float('inf'))
        else:
            scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)  # [b, b]
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size):
    """
    words_emb(query): batch x nef x seq_len; img_features(context): batch x nef x 17 x 17
    """
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)  #[batch]
            mask[i] = 0
            masks.append(mask.reshape((1, -1))) # [b, 1]
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 256, 18]
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)  # [b, 256, 18]
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num  # [b, 256, 18]
            context: batch x nef x 17 x 17  # [b, 256, 17, 17]
            weiContext: batch x nef x words_num  # [b, 256, 18]
            attn: batch x words_num x 17 x 17  # [b, 18, 17, 17]
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1) #
        att_maps.append(attn[i].unsqueeze(0).contiguous())  # [[1, 18, 17, 17], ...]
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()  # [b, 18, 256]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [b, 18, 256]
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1) # [288=b*18, 256]
        weiContext = weiContext.view(batch_size * words_num, -1) # [288=b*18, 256]
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext) # [288]
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num) # [b, 18]

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) # [b, 1]
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)  # list 长度16的 [[16, 1]]
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda() # [16, 16]

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3 # *10.0
    if class_ids is not None:
        if server:
            similarities.data.masked_fill_(masks.bool(), -float('inf'))
        else:
            similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)  # [b, b]
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels [batch] # out:[1]
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps
def words_reweight_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size, attn_map):
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)  #[batch]
            mask[i] = 0
            masks.append(mask.reshape((1, -1))) # [b, 1]
        # Get the i-th text description
        words_num = cap_lens[i]

        #####*****************************
        thresh = 2. / float(words_num)
        conf_score = []
        attn = attn_map[i]  # [1,22,128,128]
        for j in range(words_num):
            one_map = attn[j]#.cpu()  # [b,22,128,128]
            mask0 = (one_map > (2. * thresh)).float()# .cuda()  # [128, 128]
            multi = one_map * mask0
            # print('torch.sum(mask0)', torch.sum(mask0).detach().cpu().numpy())
            # print('multi', multi.detach().cpu().numpy().max())
            sum0 = torch.sum(multi)
            conf_score.append(sum0.detach().cpu().numpy()) # [19,]  # sorted_indices = np.argsort(conf_score)[::-1]  # [19]
        conf_score = np.array(conf_score)
        conf_score = torch.tensor(conf_score).cuda().view([1, 1, words_num])  # [18,]
        conf_score_sort, conf_index = conf_score.sort(descending=True)

        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 256, 18]
        word = word * conf_score  # [1, 256, 18]
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)  # [b, 256, 18]
        # batch x nef x 17*17
        context = img_features
        """
            word(query): batch x nef x words_num  # [b, 256, 18]
            context: batch x nef x 17 x 17  # [b, 256, 17, 17]
            weiContext: batch x nef x words_num  # [b, 256, 18]
            attn: batch x words_num x 17 x 17  # [b, 18, 17, 17]
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1) #
        att_maps.append(attn[i].unsqueeze(0).contiguous())  # [[1, 18, 17, 17], ...]

        #  cap_lens[i] = topK
        #  # 存起来原始的19个图像，根据排序列表重新排序attention map
        #  for j in range(num_attn):
        #      idx = sorted_indices[j]
        #      row_new.append(row[idx])
        # # 只取前5(topK)个
        #  row = np.concatenate(row_new[:topK], 1)
        #************************

        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()  # [b, 18, 256]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [b, 18, 256]
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1) # [288=b*18, 256]
        weiContext = weiContext.view(batch_size * words_num, -1) # [288=b*18, 256]

        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext) # [288]
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num) # [b, 18]

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) # [b, 1]
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)  # list 长度16的 [[16, 1]]
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda() # [16, 16]

    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3 # *10.0
    if class_ids is not None:
        if server:
            similarities.data.masked_fill_(masks.bool(), -float('inf'))
        else:
            similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)  # [b, b]
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels [batch] # out:[1]
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps

def sent_similarity(cnn_code, rnn_code, eps=1e-8):
    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)  # [1, b, 256]
        rnn_code = rnn_code.unsqueeze(0)  # [1, b, 256]

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True) # [1, b, 1]
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))  # [1, b, b]
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))  # [1, b, b]
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3  # [1, b, b]

    # --> batch_size x batch_size
    scores0 = scores0.squeeze(dim=0)  # scores0.squeeze()  # [b, b]
    return scores0
def words_similarity(img_features, words_emb, cap_lens, batch_size):
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 256, 18]
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)  # [b, 256, 18]
        # batch x nef x 17*17
        context = img_features

        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1) #
        att_maps.append(attn[i].unsqueeze(0).contiguous())  # [[1, 18, 17, 17], ...]
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()  # [b, 18, 256]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [b, 18, 256]
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1) # [288=b*18, 256]
        weiContext = weiContext.view(batch_size * words_num, -1) # [288=b*18, 256]

        row_sim = cosine_similarity(word, weiContext) # [288]
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num) # [b, 18]

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) # [b, 1]
        row_sim = torch.log(row_sim)
        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)
    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)  # list 长度16的 [[16, 1]]
    return similarities

# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCEWithLogitsLoss()(cond_real_logits, real_labels) # nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCEWithLogitsLoss()(cond_fake_logits, fake_labels) # nn.BCELoss()(cond_fake_logits, fake_labels)#
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCEWithLogitsLoss()(cond_wrong_logits, fake_labels[1:batch_size]) # nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size]) #

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCEWithLogitsLoss()(real_logits, real_labels)
        fake_errD = nn.BCEWithLogitsLoss()(fake_logits, fake_labels)
        # real_errD = nn.BCELoss()(real_logits, real_labels)
        # fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD

def generator_re_weight_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels, cap_lens, class_ids, w=1, s=1, g=1, attn=None):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCEWithLogitsLoss()(cond_logits, real_labels) # nn.BCELoss()(cond_logits, real_labels)#
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCEWithLogitsLoss()(logits, real_labels) # nn.BCELoss()(logits, real_labels) #
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        g_loss = g * g_loss
        errG_total += g_loss
        logs += 'g_loss%d: %.5f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, attnmap = words_reweight_loss(region_features, words_embs,
                                             match_labels, cap_lens, class_ids, batch_size, attn)
            w_loss = w *(w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA # lambda = 1 or 5

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
            s_loss = s * (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.5f s_loss: %.5f ' % (w_loss.item(), s_loss.item())# % (w_loss.item(), s_loss.item())#
    return errG_total, logs
def generator_loss(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels, cap_lens, class_ids, w=1, s=1, g=1):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        ### ！！！！！！ BCEWithLogitsLoss!! 包括sigmoid激活函数！！！！
        cond_errG = nn.BCEWithLogitsLoss()(cond_logits, real_labels) # nn.BCELoss()(cond_logits, real_labels)#
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCEWithLogitsLoss()(logits, real_labels) # nn.BCELoss()(logits, real_labels) #
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        g_loss = g * g_loss
        errG_total += g_loss
        # err_img = errG_total.item()
        logs += 'g_loss%d: %.5f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # region_features, words_features: batch_size x nef x 17 x 17 [b, 256, 17, 17]
            # sent_code: batch_size x nef；attnmap [batch, 1, 14, 17, 17]
            # cnn_code [b, 256] words_embs[b, 256, 18] sent_emb [b, 256]
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, attnmap = words_loss(region_features, words_embs,
                                             match_labels, cap_lens, class_ids, batch_size)
            w_loss = w *(w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA # lambda = 1 or 5
            # err_words = err_words + w_loss.item()

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
            s_loss = s * (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.item()

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.5f s_loss: %.5f ' % (w_loss.item(), s_loss.item())# % (w_loss.item(), s_loss.item())#
    return errG_total, logs
def generator_loss_oneim(netsD, image_encoder, fake_imgs, real_labels,
                   words_embs, sent_emb, match_labels, cap_lens, class_ids):
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    features = netsD[-1](fake_imgs[-1])
    cond_logits = netsD[-1].COND_DNET(features, sent_emb)
    cond_errG = nn.BCEWithLogitsLoss()(cond_logits, real_labels)
    if netsD[-1].UNCOND_DNET is not None:
        logits = netsD[-1].UNCOND_DNET(features)
        errG = nn.BCEWithLogitsLoss()(logits, real_labels)
        g_loss = errG + cond_errG
    else:
        g_loss = cond_errG
    errG_total += g_loss
    logs += 'g_lossfine: %.2f ' % (g_loss.item())

    # Ranking loss
    # words_features: batch_size x nef x 17 x 17
    # sent_code: batch_size x nef
    region_features, cnn_code = image_encoder(fake_imgs[-1])
    w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size)
    w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
    # err_words = err_words + w_loss.item()

    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
    # err_sent = err_sent + s_loss.item()

    errG_total += w_loss + s_loss
    logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
    return errG_total, logs


#### *********************** Original D/GAN loss *****************************
def discriminator_lossor(netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    return errD


def generator_lossor(netsD, image_encoder, fake_imgs, real_labels, words_embs, sent_emb,
                     match_labels, cap_lens, class_ids, w=1, s=1, g=1):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g * g_loss
        # err_img = errG_total.item()
        logs += 'g_loss%d: %.2f ' % (i, g * g_loss.item())

        # Ranking loss
        if (s != 0) and (w != 0):
            if i == (numDs - 1):
                # words_features: batch_size x nef x 17 x 17
                # sent_code: batch_size x nef
                region_features, cnn_code = image_encoder(fake_imgs[i])
                w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                                 match_labels, cap_lens, class_ids, batch_size)
                w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
                # err_words = err_words + w_loss.item()
        
                s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                             match_labels, class_ids, batch_size)
                s_loss = (s_loss0 + s_loss1) * \
                         cfg.TRAIN.SMOOTH.LAMBDA
                # err_sent = err_sent + s_loss.item()
        
                errG_total += w * w_loss + s * s_loss
                logs += 'w_loss: %.2f s_loss: %.2f ' % (w * w_loss.item(), w * s_loss.item())
        
    return errG_total, logs
### ********************************************************************************

def generator_lossor_wordsentOnly(image_encoder, fake_imgs, words_embs, sent_emb,
                     match_labels, cap_lens, class_ids, w=1, s=1):
    numDs = len(fake_imgs)
    batch_size = fake_imgs[0].size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):  # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                           match_labels, cap_lens, class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                          match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w * w_loss + s * s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w * w_loss.item(), w * s_loss.item())
    return errG_total, logs


def generator_lossor_nowordsent(netsD, fake_imgs, real_labels, sent_emb, g=1):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g * g_loss
        # err_img = errG_total.item()
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

    return errG_total, logs


def generator_lossorface(netsD, image_encoder, fake_imgs, real_labels, words_embs, sent_emb, match_labels, cap_lens, class_ids, w=1, s=1, g=1):
    numDs = len(netsD)
    if g ==1:
        g = np.ones([numDs])
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g[i] * g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):  # words_features: batch_size x nef x 17 x 17 # sent_code: batch_size x nef
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w * w_loss + s * s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss.item(), s_loss.item())
    return errG_total, logs

# ################## Loss for G and Ds ##############################
def discriminator_lossMani(netD, real_imgs, fake_imgs, conditions,
                           real_labels, fake_labels, words_embs, cap_lens, image_encoder, class_ids):
                        # w_words_embs, wrong_caps_len, wrong_cls_id
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.

    region_features_real, cnn_code_real = image_encoder(real_imgs)

    real_result = word_level_correlation(region_features_real, words_embs, cap_lens, batch_size, class_ids, real_labels)

    # w_real = word_level_correlation(region_features_real, w_words_embs, wrong_caps_len, batch_size, wrong_cls_id, fake_labels)
    # errD += (real_result + w_real) / 2.
    errD += real_result

    return errD
def generator_lossMani(netsD, image_encoder, fake_imgs, real_labels,
                       words_embs, sent_emb, match_labels, cap_lens, class_ids, VGG, real_imgs):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    feature_loss = 0
    ## numDs: 3
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCELoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss)

        # Ranking loss
        if i == (numDs - 1):
            region_features, cnn_code = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs, match_labels, cap_lens, class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)

        fake_img = fake_imgs[i]
        real_img = real_imgs[i]

        real_features = VGG(real_img)
        fake_features = VGG(fake_img)

        for i in range(len(real_features)):
            cur_real_features = real_features[i]
            cur_fake_features = fake_features[i]
            feature_loss += F.mse_loss(cur_real_features, cur_fake_features)

    errG_total += feature_loss / 3.
    logs += 'VGG feature_loss: %.2f ' % (feature_loss / 3.)

    return errG_total, logs
def DCM_generator_loss(netD, image_encoder, fake_img, real_labels,
                   words_embs, sent_emb, match_labels, cap_lens, class_ids, VGG, real_img):
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errC_total = 0

    features = netD(fake_img)
    cond_logits = netD.COND_DNET(features, sent_emb)
    cond_errG = nn.BCELoss()(cond_logits, real_labels)
    if netD.UNCOND_DNET is  not None:
        logits = netD.UNCOND_DNET(features)
        errG = nn.BCELoss()(logits, real_labels)
        g_loss = errG + cond_errG
    else:
        g_loss = cond_errG
    errC_total += g_loss
    logs += 'g_loss: %.2f ' % (g_loss)

    region_features, cnn_code = image_encoder(fake_img)
    w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
    w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

    s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb, match_labels, class_ids, batch_size)
    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

    errC_total += w_loss + s_loss
    logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)

    ### ##******* ??????????????????????? **********
    # feature_loss = F.l1_loss(fake_img, real_img) * -1
    # logs += 'feature_loss: %.2f ' % (feature_loss)
    # errC_total += feature_loss

    return errC_total, logs

def word_level_correlation(img_features, words_emb, cap_lens, batch_size, class_ids, labels):
    masks = []
    cap_lens = cap_lens.data.tolist()
    similar_list = []
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()

        context = img_features[i, :, :, :].unsqueeze(0).contiguous()

        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)

        aver = torch.mean(word, 2)
        averT = aver.unsqueeze(1)
        res_word = torch.bmm(averT, word)
        res_softmax = F.softmax(res_word, 2)  # self attention

        res_softmax = res_softmax.repeat(1, weiContext.size(1), 1)

        self_weiContext = weiContext * res_softmax

        word = word.transpose(1, 2).contiguous()
        self_weiContext = self_weiContext.transpose(1, 2).contiguous()
        word = word.view(words_num, -1)
        self_weiContext = self_weiContext.view(words_num, -1)

        row_sim = cosine_similarity(word, self_weiContext)
        row_sim = row_sim.view(1, words_num)

        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)
        similar_list.append(F.sigmoid(row_sim[0, 0]))

    similar_list = torch.tensor(similar_list, requires_grad=False).cuda()
    result = nn.BCELoss()(similar_list, labels)

    return result

def generator_lossor_Uncond(netsD, fake_imgs, real_labels):
    numDs = len(netsD)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features = netsD[i](fake_imgs[i])
        logits = netsD[i].UNCOND_DNET(features)
        errG = nn.BCELoss()(logits, real_labels)
        g_loss = errG
        errG_total += g_loss
        logs += 'Un-Cond g_loss%d: %.2f ' % (i, g_loss.item())
    return errG_total, logs


def discriminator_lossor_Uncond(netD, real_imgs, fake_imgs, real_labels, fake_labels):  # non condition GAN
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    real_logits = netD.UNCOND_DNET(real_features)
    fake_logits = netD.UNCOND_DNET(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = real_errD + fake_errD
    
    return errD


def MSE(fake, label):
    mseloss = 0
    for i in range(len(fake)):
        mseloss += nn.MSELoss()(fake[i], label[i])  # 7.6
        # mseloss += pow((fake[i] - label[i]), 2).sum()/(fake[i].shape[2]*fake[i].shape[3]*fake[i].shape[0]*fake[i].shape[1])  # 10.86  #
    return mseloss
def CycleMSE(fakeSR, Real_LR):
    mseloss = 0
    for i in range(len(fakeSR)):
        fakeLR = F.interpolate(fakeSR[i], size=[Real_LR.shape[2], Real_LR.shape[3]], mode="bicubic")
        mseloss += nn.MSELoss()(fakeLR, Real_LR)
    return mseloss

def weight_MSE(fake, label, weight):
    mseloss = 0
    for i in range(len(fake)):
        # mse1 = nn.MSELoss()(fake[i], label[i])  # 1 dim  0.1757
        w = weight[i].max(dim=1, keepdim=True)[0]  # attention map已经沿着dim 1 softmax, 求和为1矩阵
        wups = nn.Upsample(size=(fake[i].shape[2], fake[i].shape[3]))(w)
        # print(wups.max(), wups.min()) # tensor(0.6599,  tensor(0.0670
        l2 = (weight[i].shape[1]*wups)*pow((fake[i] - label[i]), 2)
        mse = l2.sum()/(fake[i].shape[2]*fake[i].shape[3]*fake[i].shape[0]*fake[i].shape[1])  # 0.6263
        if i == len(fake)-1:
            wlast = wups
        mseloss += mse
    return mseloss, wlast
##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


### ESRGAN ###
def ESRGAN_loss(netsD, fake_imgs, realim, real_labels):
    logs = ''
    errG_total = 0
    # logits = netsD(fake_imgs)
    # errG = nn.BCEWithLogitsLoss()(logits, real_labels)
    
    # Adversarial loss (relativistic average GAN)
    pred_real = netsD(fake_imgs.detach())
    pred_fake = netsD(realim)
    errG = nn.BCEWithLogitsLoss().cuda()(pred_fake - pred_real, real_labels)

    g_loss = errG
    errG_total += g_loss
    logs += 'g_loss%d: %.2f ' % (0, g_loss.item())

    return errG_total, logs


def ESRGAND_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels):
    # real_logits = netD(real_imgs)
    # fake_logits = netD(fake_imgs.detach())
    #
    # ## ESRGAN 没有激活函数结尾
    # real_errD = nn.BCEWithLogitsLoss()(real_logits, real_labels)
    # fake_errD = nn.BCEWithLogitsLoss()(fake_logits, fake_labels)
    # errD = real_errD + fake_errD
    
    pred_fake = netD(fake_imgs.detach())
    pred_real = netD(real_imgs)
    # Adversarial loss for real and fake images (relativistic average GAN)
    loss_real = nn.BCEWithLogitsLoss().cuda()(pred_real - pred_fake.mean(0, keepdim=True), real_labels)
    loss_fake = nn.BCEWithLogitsLoss().cuda()(pred_fake - pred_real.mean(0, keepdim=True), fake_labels)
    # Total loss
    errD = (loss_real + loss_fake) / 2
    
    return errD


def generator_EGAN_loss(netsD, fake_imgs, realim, real_labels):
    logs = ''
    errG_total = 0

    # Adversarial loss (relativistic average GAN)
    features = netsD(fake_imgs.detach())
    pred_fake = netsD.UNCOND_DNET(features)
    featuresr = netsD(realim)
    pred_real = netsD.UNCOND_DNET(featuresr)
    
    errG = nn.BCELoss().cuda()(pred_fake - pred_real, real_labels)
    
    g_loss = errG
    errG_total += g_loss
    logs += 'g_loss%d: %.2f ' % (0, g_loss.item())
    
    return errG_total, logs


def discriminator_EGAND_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels):
    features = netD(fake_imgs.detach())
    pred_fake = netD.UNCOND_DNET(features)
    featuresr = netD(real_imgs)
    pred_real = netD.UNCOND_DNET(featuresr)
    
    # Adversarial loss for real and fake images (relativistic average GAN)
    loss_real = nn.BCELoss().cuda()(pred_real - pred_fake.mean(0, keepdim=True), real_labels)
    loss_fake = nn.BCELoss().cuda()(pred_fake - pred_real.mean(0, keepdim=True), fake_labels)
    # Total loss
    errD = (loss_real + loss_fake) / 2
    
    return errD
