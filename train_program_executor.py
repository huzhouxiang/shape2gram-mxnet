from __future__ import print_function

import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd,autograd,init
from mxnet.gluon import Trainer,data as gdata,loss as gloss
from d2l import mxnet as d2l
from IPython import display
import matplotlib.pyplot as plt

from dataset import PartPrimitive
from model import RenderNet
from criterion import BatchIoU
from misc import clip_gradient,gather,scatter_numpy
from programs.label_config import max_param, stop_id
from options import options_train_executor



def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch >= np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        optimizer.set_learning_rate(new_lr)


def train(epoch, train_loader, model,loss, optimizer, opt,ctx,train_loss,train_iou):
    """
    one epoch training for program executor
    """
    loss_sum,iou_sum,n = 0.0,0.0,0
    for idx, data in enumerate(train_loader):
        start_t = time.time()

        shape, label, param = data
        bsz = shape.shape[0]
        n_step = label.shape[1]
        #print("label.shape:",label)
        #print("n_step:",n_step,"bsz:",bsz,"stop_id:",stop_id)
        
        index = np.array(list(map(lambda x: n_step, label)))-1
        #index = label
        
        # add noise during training, making the executor accept
        # continuous output from program generator
        label = label.reshape(-1,1).asnumpy()
        pgm_vector = 0.2 * np.random.uniform(0,1,(bsz * n_step, stop_id))
        pgm_noise = 0.2 *np.random.uniform(0,1,label.shape)
        pgm_value = 1 - pgm_noise
        #print('pgm_val.shape:',pgm_value.shape,'label.shape:',label.shape,'label.shape:',label.shape)
        pgm_vector = scatter_numpy(pgm_vector,1,label,pgm_value).reshape(bsz,n_step,stop_id)
        
        
        param_noise = nd.random_uniform(0,1,shape=param.shape)
        param_vector = param + 0.6 * (param_noise - 0.5)
        #print("param_vector.shape:",param_vector.shape)
        gt = shape.as_in_context(ctx)
        #print(pgm_vector.dtype)
        index = nd.from_numpy(index).astype('int64').as_in_context(ctx)
        pgm_vector = nd.from_numpy(pgm_vector).astype('float32').as_in_context(ctx)
        param_vector = param_vector.as_in_context(ctx)


        with autograd.record():
            pred = model(pgm_vector, param_vector, index)
            scores = nd.log_softmax(pred,axis=1)
            pred0 = scores[:,0].squeeze()*opt.n_weight
            pred1 = scores[:,1].squeeze()*opt.p_weight
            l = -nd.where(gt, pred1, pred0).mean((1,2,3))
            #l = -(nd.pick(scores1, gt, axis=1, keepdims=True)*opt.n_weight
            #    +nd.pick(scores2,(1-gt), axis=1, keepdims=True)*opt.p_weight).mean((1,2,3,4))
        l.backward()
                                        
        #clip_gradient(optimizer, opt.grad_clip)
        #optimizer._allreduce_grads();

        optimizer.step(l.shape[0],ignore_stale_grad=True)
        
        l = l.mean().asscalar()
        
        pred = nd.softmax(pred,axis = 1)
        pred = pred[:, 1, :, :, :]
        s1 = gt.reshape(-1, 32, 32, 32).astype('float32').as_in_context(mx.cpu())
        s2 = pred.squeeze().as_in_context(mx.cpu())
        #print(s2.shape)
        s2 = (s2 > 0.5)

        batch_iou = BatchIoU(s1, s2)
        iou = batch_iou.mean()
        end_t = time.time()
        loss_sum+=l
        n+=1
        iou_sum+=iou

        if idx % (opt.info_interval * 10) == 0:
            print("Train: epoch {} batch {}/{}, loss13 = {:.3f}, iou = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(train_loader), l, iou, end_t - start_t))
            sys.stdout.flush()
        
    train_loss.append(loss_sum/n)
    train_iou.append(iou_sum/n)


def validate(epoch, val_loader, model, loss, opt, ctx,val_loss,val_iou, gen_shape=False):

    # load pre-fixed randomization
    try:
        rand1 = np.load(opt.rand1)
        rand2 = np.load(opt.rand2)
        rand3 = np.load(opt.rand3)
    except:
        rand1 = np.random.rand(opt.batch_size * opt.seq_length, stop_id).astype(np.float32)
        rand2 = np.random.rand(opt.batch_size * opt.seq_length, 1).astype(np.float32)
        rand3 = np.random.rand(opt.batch_size, opt.seq_length, max_param - 1).astype(np.float32)
        np.save(opt.rand1, rand1)
        np.save(opt.rand2, rand2)
        np.save(opt.rand3, rand3)

    generated_shapes = None
    original_shapes = None
    
    loss_sum,iou_sum,n = 0.0,0.0,0
    for idx, data in enumerate(val_loader):
        start_t = time.time()

        shape, label, param = data

        bsz = shape.shape[0]
        n_step = label.shape[1]
        index = np.array(list(map(lambda x: n_step, label)))
        index = index - 1

        # add noise during training, making the executor accept
        # continuous output from program generator
        
        label = label.reshape(-1,1).asnumpy()
        pgm_vector = 0.1*rand1
        pgm_noise = 0.1*rand2
        pgm_value = np.ones(label.shape) - pgm_noise
        #print('pgm_val.shape:',pgm_value.shape,'label.shape:',label.shape,'label.shape:',label.shape)
        pgm_vector = scatter_numpy(pgm_vector,1,label,pgm_value).reshape(bsz,n_step,stop_id)

        param_noise = nd.from_numpy(rand3)
        #print(param.shape,param_noise.shape)
        param_vector = param + 0.6 * (param_noise - 0.5)
        
        
        gt = shape.astype('float32').as_in_context(ctx)
        index = nd.from_numpy(index).astype('int64').as_in_context(ctx)
        pgm_vector = nd.from_numpy(pgm_vector).as_in_context(ctx)
        param_vector = param_vector.as_in_context(ctx)
        #prediction
        pred = model(pgm_vector, param_vector, index)
        scores = nd.log_softmax(pred,axis=1)
        pred0 = scores[:,0].squeeze()*opt.p_weight
        pred1 = scores[:,1].squeeze()*opt.n_weight
        l = -nd.where(gt, pred1, pred0).mean((1,2,3))
        #print(pred2.dtype,gt.dtype)
        #l = loss(pred,gt,sample_weight = nd.array([opt.n_weight,opt.p_weight]))

        l = l.mean().asscalar()
        pred = nd.softmax(pred,axis=1)
        pred = pred[:, 1, :, :, :]
        s1 = gt.reshape(-1, 32, 32, 32).as_in_context(mx.cpu())
        s2 = pred.squeeze().as_in_context(mx.cpu())
        s2 = (s2 > 0.5)

        batch_iou = BatchIoU(s1, s2)
        iou = batch_iou.mean()
        loss_sum+=l
        n+=1
        iou_sum+=iou
        
        if(idx+1)%5==0 and gen_shape:
            if original_shapes is None:
                original_shapes = s1.expand_dims(axis=0)
                generated_shapes = s2.expand_dims(axis=0)
            else:
                original_shapes = nd.concat(original_shapes,s1.expand_dims(axis=0),dim=0)
                generated_shapes = nd.concat(generated_shapes,s2.expand_dims(axis=0),dim=0)
        end_t = time.time()

        if (idx + 1) % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, loss13 = {:.3f}, iou = {:.3f}, time = {:.3f}"
                  .format(epoch, idx + 1, len(val_loader), l, iou, end_t - start_t))
            sys.stdout.flush()
        if(idx+1>len(val_loader)/10):
            
            break;     
    val_loss.append(loss_sum/n)
    val_iou.append(iou_sum/n)

    return generated_shapes, original_shapes





def run():
 
    opt = options_train_executor.parse()

    print('===== arguments: program executor =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: program executor =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)


    # build dataloader
    train_loader = gdata.DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    val_set = PartPrimitive(opt.val_file)
    val_loader = gdata.DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )

    # build the model
    ctx = d2l.try_gpu()                   
    model = RenderNet(opt)
    model.initialize(init = init.Xavier(),ctx = ctx)                                        

    loss = gloss.SoftmaxCrossEntropyLoss(axis = 1,weight = 5)

    optimizer = Trainer(model.collect_params(),"adam",
                        {"learning_rate":opt.learning_rate,"wd":opt.weight_decay,
                         'beta1':opt.beta1, 'beta2':opt.beta2})
    train_from0 = False;
    if train_from0:
        if os.path.exists('./model of executor'):
            model.load_parameters('model of executor')
            print("loaded parameter of model")
        if os.path.exists('./optimizer of executor'):
            optimizer.load_states('optimizer of executor')
            print("loaded state of trainer")
            
    for epoch in range(1, opt.epochs+1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("###################")
        print("training")
        train(epoch, train_loader, model,loss,optimizer, opt,ctx,train_loss,train_iou)

        print("###################")

        print("testing")
        '''
        gen_shapes, ori_shapes = validate(epoch, val_loader, model,
                                          loss, opt,ctx, val_loss,val_iou, gen_shape=True)
        gen_shapes = (gen_shapes > 0.5)
        gen_shapes = gen_shapes.astype(np.float32)
        iou = BatchIoU(ori_shapes, gen_shapes)
        print("Mean IoU: {:.3f}".format(iou.mean().asscalar()))
        '''
        if epoch % opt.save_interval == 0:
            print('Saving...')
            optimizer.save_states("optimizer of executor_3"),
            model.save_parameters("model of executor_3")


    print('Saving...')
    optimizer.save_states("optimizer of executor_3"),
    model.save_parameters("model of executor_3")


if __name__ == '__main__':
    run()
