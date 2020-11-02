from __future__ import print_function

import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd,autograd,init
from mxnet.gluon import Trainer,data as gdata,loss as gloss
from d2l import mxnet as d2l

from dataset import Synthesis3D
from model import BlockOuterNet
from criterion import LSTMClassCriterion, LSTMRegressCriterion,BatchIoU
from misc import clip_gradient, decode_to_shape_new,gather,decode_multiple_block
from options import options_train_generator


def train(epoch, train_loader, model, crit_cls, crit_reg, optimizer, opt,ctx):
    """
    One epoch training
    """
    cls_w = opt.cls_weight
    reg_w = opt.reg_weight
    # the prob: > 1
    # the input of step t Operator where is missing FInferType attributeis always sampled from the output of step t-1
    sample_prob = opt.inner_sample_prob

    for idx, data in enumerate(train_loader):
        start = time.time()
        #data, pgm, pgm_mask, param, param_mask
        shapes, labels, masks, params, param_masks = data[0], data[1], data[2], data[3], data[4]
        gt = shapes
        shapes = nd.expand_dims(shapes, axis = 1)
        #print(labels[0],params[0])
        shapes = shapes.as_in_context(ctx)
        
        labels = labels.as_in_context(ctx)
        labels2 = labels.as_in_context(ctx)
        
        masks = masks.as_in_context(ctx)
        params = params.as_in_context(ctx)
        param_masks = param_masks.as_in_context(ctx)
        
        
        #shapes.attach_grad(),labels.attach_grad()
        with autograd.record():
            out = model(shapes, labels, sample_prob)
            #out = model.decode(shapes)
        
            # reshape
            bsz, n_block, n_step = labels.shape
            labels = labels.reshape(bsz, -1)
            masks = masks.reshape(bsz, -1)
            out_pgm = out[0].reshape(bsz, n_block * n_step, opt.program_size + 1)
            
            bsz, n_block, n_step, n_param = params.shape
            params = params.reshape(bsz, n_block * n_step, n_param)
            param_masks = param_masks.reshape(bsz, n_block * n_step, n_param)
            out_param = out[1].reshape(bsz, n_block * n_step, n_param)
            
            loss_cls, acc = crit_cls(out_pgm, labels, masks)
            loss_reg = crit_reg(out_param, params, param_masks)
            loss = cls_w*loss_cls+reg_w*loss_reg
        loss.backward()

        optimizer.step(bsz,ignore_stale_grad=True)
        
        loss_cls = loss_cls.mean().asscalar()
        loss_reg = loss_reg.mean().asscalar()
        
        end = time.time()
        
        
        if idx % (opt.info_interval*10) == 0:
            out_1 = nd.round(out[0]).astype('int64')
            out_2 =nd.round(out[1]).astype('int64')
            pred = nd.from_numpy(decode_multiple_block(out_1, out_2)).astype("float32").as_in_context(mx.cpu())
            IoU = BatchIoU(pred,gt)
            print("Train: epoch {} batch {}/{},loss_cls = {:.3f},loss_reg = {:.3f},acc = {:.3f},IoU = {:.3f},time = {:.2f}"
                  .format(epoch, idx, len(train_loader), loss_cls, loss_reg, acc[0].asscalar(), IoU.mean(),end - start))
            sys.stdout.flush()
        

def validate(epoch, val_loader, model, crit_cls, crit_reg, opt,ctx, gen_shape=False):
    """
    One validation
    """
    generated_shapes = []
    original_shapes = []
    sample_prob = opt.inner_sample_prob
    loss_cls_sum,loss_reg_sum,n = 0.0,0.0,0
    
    for idx, data in enumerate(val_loader):
        start = time.time()

        shapes, labels, masks, params, param_masks = data[0], data[1], data[2], data[3], data[4]
        gt = shapes
        shapes = nd.expand_dims(shapes, axis = 1)

        shapes = shapes.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        masks = masks.as_in_context(ctx)
        params = params.as_in_context(ctx)
        param_masks = param_masks.as_in_context(ctx)
        with autograd.train_mode():     
            out = model.decode(shapes)
        #out = model(shapes, labels, sample_prob)
        bsz, n_block, n_step = labels.shape
        labels = labels.reshape(bsz, n_block * n_step)
        masks = masks.reshape(bsz, n_block * n_step)
        out_pgm = out[0].reshape(bsz, n_block * n_step, opt.program_size + 1)

        bsz, n_block, n_step, n_param = params.shape
        params = params.reshape(bsz, n_block * n_step, n_param)
        param_masks = param_masks.reshape(bsz, n_block * n_step, n_param)
        out_param = out[1].reshape(bsz, n_block * n_step, n_param)
        loss_cls, acc = crit_cls(out_pgm, labels, masks)
        loss_reg = crit_reg(out_param, params, param_masks)
   
        end = time.time()
        
        
        loss_cls = loss_cls.mean().asscalar()
        loss_reg = loss_reg.mean().asscalar()
        
        
        if idx % opt.info_interval == 0:
            out_1 = nd.round(out[0]).astype('int64')
            out_2 = nd.round(out[1]).astype('int64')
            pred = nd.from_numpy(decode_multiple_block(out_1, out_2)).astype("float32").as_in_context(mx.cpu())
            IoU = BatchIoU(pred,gt)
            print("Test: epoch {} batch {}/{}, loss_cls = {:.3f}, loss_reg = {:.3f}, acc = {:.3f}, IoU = {:.3f} time = {:.3f}"
                  .format(epoch, idx, len(val_loader), loss_cls, loss_reg, acc[0].asscalar(), IoU.mean(), end - start))
            sys.stdout.flush()


def run():

    opt = options_train_generator.parse()

    print('===== arguments: program generator =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: program generator =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)



        # build model
    ctx = d2l.try_gpu()
    model = BlockOuterNet(opt)
    model.init_blocks(ctx)

    crit_cls = LSTMClassCriterion()
    crit_reg = LSTMRegressCriterion()
    ctri_cls = crit_cls.initialize(ctx = ctx)
    ctri_reg = crit_reg.initialize(ctx = ctx)
    
    optimizer = Trainer(model.collect_params(),"adam",
                    {"learning_rate":opt.learning_rate,"wd":opt.weight_decay,
                     'beta1':opt.beta1, 'beta2':opt.beta2, 'clip_gradient': opt.grad_clip})
                           

    # build dataloader
    train_set = Synthesis3D(opt.train_file, n_block=opt.outer_seq_length)
    train_loader = gdata.DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_set = Synthesis3D(opt.val_file, n_block=opt.outer_seq_length)
    val_loader = gdata.DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    for epoch in range(1, opt.epochs+1):
        print("###################")
        print("training")

        train(epoch, train_loader, model, crit_cls, crit_reg, optimizer, opt,ctx)

        print("###################")
        print("testing")
        validate(epoch, val_loader, model, crit_cls, crit_reg, opt,ctx,True)
        if epoch % 1 == 0:
            print('Saving...')
            optimizer.save_states("optimizer of PG"),
            model.save_parameters("model of blockouternet")

    optimizer.save_states("optimizer of PG"),
    model.save_parameters("model of blockouternet")


if __name__ == '__main__':
    run()