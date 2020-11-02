from __future__ import print_function

import numpy as np
import mxnet
from mxnet import gluon,nd
from mxnet.gluon import nn
from misc import to_contiguous,gather

class LSTMClassCriterion(nn.Block):
    def __init__(self,**kwargs):
        super(LSTMClassCriterion, self).__init__(**kwargs)

    def forward(self, pred, target, mask):
        # truncate to the same size
        pred = pred.copy()
        bsz = pred.shape[0]
        target = target.copy()
        mask = mask.copy()
        #print('target.shape:',target.shape)                         #target.shape: (8, 30)
        target = target[:, :pred.shape[1]].reshape(-1, 1)
        #print('target.shape:',target.shape)                         #target.shape: (240, 1)
        mask = mask[:, :pred.shape[1]].reshape(-1, 1)
        #print('mask.shape:',mask.shape)                             #mask.shape: (240, 1)
        pred = pred.reshape(-1, pred.shape[2])
        #print('pred.shape:',pred.shape)                             #pred.shape: (240, 22)
        # compute loss
        #target = target.expand_dims(axis=0).broadcast_to(shape=(2,target.shape[0],target.shape[1]))
        loss = - nd.pick(pred,target).expand_dims(axis=1)* mask #gather(pred,dim=1,index = target) * mask   #gather_nd
        #print("loss.shape:",loss.shape,nd.pick(pred,target).shape)
        
        loss = nd.sum(loss) / nd.sum(mask)

                                                                           
        # compute accuracy
        idx = nd.argmax(pred, axis=1).astype('int64')
        #print( idx.dtype,target.dtype)
        correct = (idx==nd.squeeze(target))
        correct = correct.astype('float32') * nd.squeeze(mask)
        accuracy = nd.sum(correct) / nd.sum(mask)
        return loss, accuracy


class LSTMRegressCriterion(nn.Block):
    def __init__(self,**kwargs):
        super(LSTMRegressCriterion, self).__init__(**kwargs)

    def forward(self, pred, target, mask):
        # truncate to the same size
        pred = pred.copy()
        target = target.copy()
        mask = mask.copy()
        target = target[:, :pred.shape[1], :]
        mask = mask[:, :pred.shape[1], :]
        # compute the loss
        diff = 0.5 * nd.square(pred - target)
        diff = diff * mask
        #print(diff.shape)
        output = nd.sum(diff) / nd.sum(mask)
        return output


def BatchIoU(s1, s2):
    """
    :param s1: first sets of shapes
    :param s2: second sets of shapes
    :return: IoU
    """
    assert s1.shape[0] == s2.shape[0], "# (shapes1, shapes2) don't match"
    v1 = nd.sum(s1 > 0.5, axis=(1, 2, 3)).asnumpy()
    v2 = nd.sum(s2 > 0.5, axis=(1, 2, 3)).asnumpy()
    I = nd.sum((s1 > 0.5) * (s2 > 0.5), axis=(1, 2, 3)).asnumpy()
    U = v1 + v2-I
    inds = U == 0
    U[inds] = 1
    I[inds] = 1
    #print("v1",v1,"\nv2:",v2,"\nI",I,"\nU",U)
    #print("U.min:",U.min(),"\nI.min():",I.min())
    IoU = I.astype(np.float32) / U.astype(np.float32)

    return IoU


def SingleIoU(s1, s2):
    """
    :param s1: shape 1
    :param s2: shape 2
    :return: Iou
    """
    v1 = np.sum(s1 > 0.5)
    v2 = np.sum(s2 > 0.5)
    I = np.sum((s1 > 0.5) * (s2 > 0.5))
    U = v1 + v2 - I
    if U == 0:
        IoU = 1
    else:
        IoU = float(I) / float(U)

    return IoU
