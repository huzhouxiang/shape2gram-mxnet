from __future__ import print_function

import sys
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd,autograd,init
from mxnet.gluon import Trainer,data as gdata,loss as gloss,nn
from d2l import mxnet as d2l

from dataset import ShapeNet3D
from model import BlockOuterNet, RenderNet
from criterion import BatchIoU
from misc import clip_gradient, decode_multiple_block
from options import options_guided_adaptation,options_train_generator,options_train_executor
from visualization.util_vtk import visualization


def train(epoch, train_loader, generator, executor, criterion, optimizer, opt,ctx):
    """
    one epoch guided adaptation
    """
    def set_bn_eval(m):
        if m.prefix[:9]=='batchnorm':
            m._kwargs['use_global_stats']=True
            m.grad_req = 'null'

    executor.apply(set_bn_eval)
    for idx, data in enumerate(train_loader):
        start = time.time()
        shapes = data.as_in_context(ctx)
        raw_shapes = data
        shapes = shapes.expand_dims(axis = 1)
        with autograd.record():
            pgms, params = generator.decode(shapes)
            
            # truly rendered shapes
            pgms_int = nd.round(pgms).astype('int64')
            params_int = nd.round(params).astype('int64')
           

            # neurally rendered shapes
            pgms = nd.exp(pgms)
            bsz, n_block, n_step, n_vocab = pgms.shape
            pgm_vector = pgms.reshape(bsz*n_block, n_step, n_vocab)
            bsz, n_block, n_step, n_param = params.shape
            param_vector = params.reshape(bsz*n_block, n_step, n_param)
            index = (n_step - 1) * nd.ones(bsz * n_block).astype('int64')
            index = index.as_in_context(ctx)
            
            pred = executor(pgm_vector, param_vector, index)
            pred = nd.softmax(pred,axis = 1)
            #print(pred.shape)
            pred = pred[:, 1]
            pred = pred.reshape(bsz, n_block, 32, 32, 32)

            rec = nd.max(pred, axis=1)
            #print("rec.shape:",rec.shape,"shapes.shape:",shapes.shape)
            #rec1 = rec.expand_dims(axis=1)
            rec1 = nd.log(rec+ 1e-11)
            rec0 = nd.log(1 - rec+1e-11)
            #rec_all = nd.concat(rec0, rec1, dim=1)
            #rec_all1 = nd.log(rec_all + 1e-10)
            #rec_all2 = nd.log(1-rec_all + 1e-10) 
            gt = shapes.squeeze().astype('int64')
            loss = -nd.where(gt,rec1,rec0).mean(axis = (1,2,3))
            #loss = -(nd.pick(rec_all1,gt,axis = 1,keepdims=True)).mean(axis = (1,2,3,4))
            #loss = criterion(rec_all, gt)
        loss.backward()
        optimizer.step(loss.shape[0],ignore_stale_grad=True)
        l = loss.mean().asscalar()
        
        rendered_shapes = decode_multiple_block(pgms_int, params_int)
        rendered_shapes = nd.from_numpy(rendered_shapes).astype('float32').as_in_context(mx.cpu())
        IoU2= BatchIoU(raw_shapes,rendered_shapes)
        reconstruction = (rec.as_in_context(mx.cpu())>0.5).astype('float32')
        IoU1 = BatchIoU(reconstruction, raw_shapes)
        #print("IoU1:",IoU1,IoU2)
        IoU1 = IoU1.mean()
        IoU2 = IoU2.mean()
        

        end = time.time()

        if idx % opt.info_interval == 0:
            print("Train: epoch {} batch {}/{}, loss = {:.3f}, IoU1 = {:.3f}, IoU2 = {:.3f}, time = {:.3f}"
                  .format(epoch, idx, len(train_loader), l, IoU1, IoU2, end - start))
            sys.stdout.flush()


def validate(epoch, val_loader, generator, opt,ctx,gen_shape=False):
    """
    evaluate program generator, in terms of IoU
    """
    generated_shapes = []
    original_shapes = []
    for idx, data in enumerate(val_loader):
        start = time.time()
        shapes = data.as_in_context(ctx)
        shapes = nd.expand_dims(shapes, axis=1)
        with autograd.train_mode():
            out = generator.decode(shapes)

        end = time.time()
        
        if gen_shape:
            out_1 = nd.round(out[0]).astype('int64')
            out_2 = nd.round(out[1]).astype('int64')
            generated_shapes.append(decode_multiple_block(out_1, out_2).astype("float32"))
            original_shapes.append(data.asnumpy())

        if idx % opt.info_interval == 0:
            print("Test: epoch {} batch {}/{}, time={:.3f}"
                  .format(epoch, idx, len(val_loader), end - start))

    if gen_shape:
        generated_shapes = np.concatenate(generated_shapes, axis=0)
        original_shapes = np.concatenate(original_shapes, axis=0)

    return generated_shapes, original_shapes


def run():
    # get options
    opt = options_guided_adaptation.parse()
    opt_gen = options_train_generator.parse()
    opt_exe = options_train_executor.parse()
    print('===== arguments: guided adaptation =====')
    for key, val in vars(opt).items():
        print("{:20} {}".format(key, val))
    print('===== arguments: guided adaptation =====')

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # build loaders
    train_set = ShapeNet3D(opt.train_file)
    train_loader = gdata.DataLoader(
        dataset=train_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    val_set = ShapeNet3D(opt.val_file)
    val_loader = gdata.DataLoader(
        dataset=val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers
    )
    def visual(path,epoch,gen_shapes,file_name,nums_samples):
        data = gen_shapes.transpose((0, 3, 2, 1))
        data = np.flip(data, axis=2)
        num_shapes = data.shape[0]
        for i in range(min(nums_samples,num_shapes)):
            voxels = data[i]
            save_name = os.path.join(path, file_name.format(epoch,i))
            visualization(voxels,
                          threshold=0.1,
                          save_name=save_name,
                          uniform_size=0.9)
    ctx = d2l.try_gpu()

    # load program generator
    generator = BlockOuterNet(opt_gen)
    generator.init_blocks(ctx)
    generator.load_parameters("model of blockouternet")

    # load program executor
    executor = RenderNet(opt_exe)
    executor.initialize(init = init.Xavier(),ctx = ctx)
    executor.load_parameters("model of executor")

    # build loss functions
    criterion = gloss.SoftmaxCrossEntropyLoss(axis=1,from_logits=True)

    optimizer = Trainer(generator.collect_params(),"adam",
                        {"learning_rate":opt.learning_rate,"wd":opt.weight_decay,
                         'beta1':opt.beta1, 'beta2':opt.beta2,'clip_gradient': opt.grad_clip})


    print("###################")
    print("testing")
    gen_shapes, ori_shapes = validate(0, val_loader, generator, opt,ctx,gen_shape=True)
    #visual('imgs of chairs/adaption/chair/',0,ori_shapes,'GT {}-{}.png',8)
    #visual('imgs of chairs/adaption/chair/',0,gen_shapes,'epoch{}-{}.png',8)

    gen_shapes = nd.from_numpy(gen_shapes)
    ori_shapes = nd.from_numpy(ori_shapes)
    #print(gen_shapes.dtype,ori_shapes.dtype)
    #print("done",ori_shapes.shape,gen_shapes.shape)


    IoU = BatchIoU(gen_shapes,ori_shapes)
    #print(IoU)
    print("iou: ", IoU.mean())


    best_iou = 0
    print(opt.epochs)
    for epoch in range(1, opt.epochs+1):
        print("###################")
        print("adaptation")
        train(epoch, train_loader, generator, executor, criterion, optimizer, opt,ctx)
        print("###################")
        print("testing")
        gen_shapes, ori_shapes = validate(epoch, val_loader, generator, opt,ctx,gen_shape=True)
        #visual('imgs of chairs/adaption/chair/',epoch,gen_shapes,'epoch{}-{}.png',8)
        gen_shapes = nd.from_numpy(gen_shapes)
        ori_shapes = nd.from_numpy(ori_shapes)
        IoU = BatchIoU(gen_shapes,ori_shapes)

        print("iou: ", IoU.mean())

        if epoch % opt.save_interval == 0:

            print('Saving...')
            generator.save_parameters("generator of GA on shapenet")
            optimizer.save_states("optimazer of generator of GA on shapenet")

        if IoU.mean() >= best_iou:
            print('Saving best model')
            generator.save_parameters("generator of GA on shapenet")
            optimizer.save_states("optimazer of generator of GA on shapenet")
            best_iou = IoU.mean()




if __name__ == '__main__':
    run()
