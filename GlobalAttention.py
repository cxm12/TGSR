# coding: utf-8
"""
Global attention takes a matrix and a query matrix.
Based on each query vector q, it computes a parameterized convex combination of the matrix
based.
H_1 H_2 H_3 ... H_n
  q   q   q       q
    |  |   |       |
      \ |   |      /
              .....
          \   |  /
                  a
Constructs a unit mapping.
$$(H_1 + H_n, q) => (a)$$
Where H is of `batch x n x dim` and q is of `batch x dim`.

References:
https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
http://www.aclweb.org/anthology/D15-1166
"""

import torch
import torch.nn as nn

server = 0


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL  [b, 256, 18]
    context: batch x ndf x ih x iw (sourceL=ihxiw)  # [b, 256, 17, 17]
    mask: batch_size x sourceL
    gamma1:determines how much attention is paid to features of its relevant sub-regions
    when computing the region-context vector for a word.
    决定了在计算单词的区域上下文向量时应多注意其相关子区域的特征。
    """
    batch_size, queryL = query.size(0), query.size(2)  # b, 18
    ih, iw = context.size(2), context.size(3)  # 17, 17
    sourceL = ih * iw  # 289
    
    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)  # [b, 256, 289]
    contextT = torch.transpose(context, 1, 2).contiguous()  # [b, 289, 256]
    
    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL [b, 289, 18]
    attn = attn.view(batch_size * sourceL, queryL)  # [4624=b*289, 18]
    attn = nn.Softmax(dim=1)(attn)  # Eq. (8)
    
    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)  # [b,289,18]
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()  # [b,18,289]
    attn = attn.view(batch_size * queryL, sourceL)  # [288, 289]
    #  Eq. (9)
    attn = attn * gamma1  # *4.0
    attn = nn.Softmax(dim=1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)  # [b, 18, 289]
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()  # [b, 289, 18]
    
    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)  # [b, 256, 18]
    
    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):  # 32, 256
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None
    
    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL
    
    def forward(self, input, context):  # [bs,16,64,64]  [bs,256,18]
        """
            input: batch x idf x ih x iw (queryL=ihxiw)    [20,32,64,64]
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        
        # --> batch x queryL x idf  [b, 32, h*w]
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1 [b, 256, 18, 1]
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL  [b, 32, 18]
        sourceT = self.conv_context(sourceT).squeeze(3)
        
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL  [b, h*w, 18]
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            if server:
                attn.data.masked_fill_(mask.data.bool(), -float('inf'))  # server
            else:
                attn.data.masked_fill_(mask.data, -float('inf'))  # desktop
                # attn.data.masked_fill_(mask.data.byte(), -float('inf')) # desktop
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()
        
        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        
        return weightedContext, attn


### 对word vector([b,f,18])计算一个权值w([b,18])，对其重新分配权值再与图片特征结合
### 效果不好
class GlobalAttentionGeneral_weight(nn.Module):
    def __init__(self, idf, cdf):  # 32, 256
        super(GlobalAttentionGeneral_weight, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None
    
    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL
    
    def forward(self, input, context):  # [bs,16,64,64]  [bs,256,18]
        """input: batch x idf x ih x iw (queryL=ihxiw)    [20,32,64,64]
           context: batch x cdf x sourceL """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        
        # --> batch x queryL x idf  [b, 32, h*w]
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1 [b, 256, 18, 1] --> batch x idf x sourceL  [b, 32, 18]
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL  [b, h*w, 18] --> batch*queryL x sourceL
        attn = torch.bmm(targetT, sourceT).view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            if server:
                attn.data.masked_fill_(mask.data.bool(), -float('inf'))  # server
            else:
                attn.data.masked_fill_(mask.data, -float('inf'))  # desktop
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL --> batch x sourceL x queryL
        attn = torch.transpose(attn.view(batch_size, queryL, sourceL), 1, 2).contiguous()
        #############================ weight ================##############
        w = attn.clone().view(batch_size, sourceL, ih, iw)
        
        w1 = nn.AvgPool2d(ih, stride=ih, padding=0)(w).view(batch_size, 1, sourceL)
        # w1 = nn.MaxPool2d(ih, stride=ih, padding=0)(w).view(batch_size, 1, sourceL)
        sourceTw = sourceT * w1
        
        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(sourceTw, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        
        return weightedContext, attn
