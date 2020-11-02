from __future__ import print_function

import numpy as np
import mxnet
import d2lzh as d2l
from mxnet import nd,init
from mxnet.gluon import nn,data as gdata,rnn
from misc import render_block,gather


class BlockOuterNet(nn.Block):
    """
    predict block-level programs and parameters
    block-LSTM
    """
    def __init__(self, opt,**kwargs):
        super(BlockOuterNet, self).__init__(**kwargs)
        self.feat_size = opt.shape_feat_size
        self.input_size = opt.outer_input_size
        self.rnn_size = opt.outer_rnn_size
        self.num_layers = opt.outer_num_layers
        self.drop_prob = opt.outer_drop_prob
        self.seq_length = opt.outer_seq_length
        self.ctx = d2l.try_gpu()
        self.shape_feat = Conv3DNet(self.feat_size, input_channel=3, power=3)
        
        self.core = rnn.LSTM(hidden_size = self.rnn_size,
                                num_layers = self.num_layers,
                                dropout = self.drop_prob,
                                input_size = self.input_size)
        self.inner_net = BlockInnerNet(opt)

    def init_blocks(self, ctx):
        self.shape_feat.initialize(init = init.Xavier(),ctx=ctx)
        self.core.initialize(ctx = ctx)
        self.inner_net.init_weights(ctx = ctx)
    def hidden_states(self,bsz,ctx):
        return (nd.zeros(shape=(self.num_layers, bsz, self.rnn_size),ctx = ctx),
                nd.zeros(shape = (self.num_layers, bsz, self.rnn_size),ctx = ctx))

    def forward(self, x, y, sample_prob=None):
        batch_size = x.shape[0]
        state = self.hidden_states(batch_size,self.ctx)

        outputs_pgm = []
        outputs_param = []

        rendered_shapes = np.zeros((batch_size, 32, 32, 32), dtype=np.float32)
        rendered_shapes.setflags(write = 1)
        def combine(x, y):
            y.setflags(write=1)
            y = nd.from_numpy(y).as_in_context(self.ctx)
            y = y.expand_dims(axis=1);
            return nd.concat(x, y, dim=1)
        #print(combine(x,rendered_shapes).shape)
        fc_feats = self.shape_feat(combine(x, rendered_shapes))

        seq = y
        #print("seq.shape",seq.shape)   #(8, 10, 3)
        for i in range(seq.shape[1]):
            if i == 0:
                xt = fc_feats
            else:
                prob_pre = nd.exp(outputs_pgm[-1])
                it1 = nd.argmax(prob_pre, axis = 2).asnumpy().astype(np.int64)
                
                #print("it1.shape:",it1.shape) #(8, 3)
                it2 = outputs_param[-1].asnumpy()
                rendered_shapes.setflags(write = 1)
                rendered_shapes = render_block(rendered_shapes, it1, it2)
                #print('rendered_shapes.shape:',rendered_shapes.shape)
                xt = self.shape_feat(combine(x, rendered_shapes))
                
            output, state = self.core(xt.expand_dims(axis = 0), state)
            #print( 'output.shape:',output.shape,y.shape)    #(1, 8, 64) (8, 10, 3)
            output = nd.relu(output)
            pgm, param = self.inner_net(output.squeeze(), y[:, i], sample_prob)
            #print('pgm.shape:',pgm.shape,'param.shape:',param.shape)
            outputs_pgm.append(pgm)
            outputs_param.append(param)
        outputs_pgm = [_.expand_dims(axis = 1) for _ in outputs_pgm]
        outputs_param = [_.expand_dims(axis = 1) for _ in outputs_param]
        pgms = outputs_pgm[0]
        params = outputs_param[0]
        for i in range(1,len(outputs_pgm)):
            pgms = nd.concat(pgms,outputs_pgm[i],dim = 1)
            params = nd.concat(params,outputs_param[i],dim = 1)

        return [pgms,params]

    def decode(self, x):
        batch_size = x.shape[0]
        state = self.hidden_states(batch_size,self.ctx)

        outputs_pgm = []
        outputs_param = []

        rendered_shapes = np.zeros((batch_size, 32, 32, 32), dtype=np.float32)
        rendered_shapes.setflags(write = 1)
        def combine(x, y):
            y.setflags(write=1)
            y = nd.from_numpy(y).as_in_context(self.ctx)
            y = y.expand_dims(axis=1);
            return nd.concat(x, y, dim=1)
        fc_feats = self.shape_feat(combine(x, rendered_shapes))

        for i in range(self.seq_length):
            if i == 0:
                xt = fc_feats
            else:
                prob_pre = nd.exp(outputs_pgm[-1])
                it1 = nd.argmax(prob_pre, axis = 2).asnumpy().astype(np.int64)
                
                it2 = outputs_param[-1].asnumpy()
                rendered_shapes.setflags(write = 1)
                rendered_shapes = render_block(rendered_shapes, it1, it2)
                xt = self.shape_feat(combine(x, rendered_shapes))
                
            output, state = self.core(xt.expand_dims(axis = 0), state)
            output = nd.relu(output)
            pgm, param = self.inner_net.decode(output.squeeze())
            outputs_pgm.append(pgm)
            outputs_param.append(param)
        outputs_pgm = [_.expand_dims(axis = 1) for _ in outputs_pgm]
        outputs_param = [_.expand_dims(axis = 1) for _ in outputs_param]
        pgms = outputs_pgm[0]
        params = outputs_param[0]
        for i in range(1,len(outputs_pgm)):
            pgms = nd.concat(pgms,outputs_pgm[i],dim = 1)
            params = nd.concat(params,outputs_param[i],dim = 1)

        return [pgms,params]

class BlockInnerNet(nn.Block):
    """
    Inner Block Net
    use last pgm as input for each time step
    step-LSTM
    """
    def __init__(self, opt):
        super(BlockInnerNet, self).__init__()

        self.vocab_size = opt.program_size
        self.max_param = opt.max_param
        self.input_size = opt.inner_input_size
        self.rnn_size = opt.inner_rnn_size
        self.num_layers = opt.inner_num_layers
        self.drop_prob = opt.inner_drop_prob
        self.seq_length = opt.inner_seq_length
        self.cls_feat_size = opt.inner_cls_feat_size
        self.reg_feat_size = opt.inner_reg_feat_size
        self.sample_prob = opt.inner_sample_prob
        self.ctx = d2l.try_gpu()
        self.pgm_embed = nn.Embedding(self.vocab_size + 1, self.input_size)
        
        self.core = rnn.LSTM(hidden_size = self.rnn_size,
                                num_layers = self.num_layers,
                                dropout = self.drop_prob,
                                input_size = self.input_size)
        
        self.logit1 = nn.Dense(self.cls_feat_size)
        self.logit2 = nn.Dense(self.vocab_size + 1)
        self.regress1 = nn.Dense(self.reg_feat_size)
        self.regress2 = nn.Dense((self.vocab_size + 1) * self.max_param)

    def init_weights(self,ctx):
        initrange = 0.1
        self.pgm_embed.initialize(init.Uniform(initrange),ctx = ctx)
        self.logit1.initialize(init.Uniform(initrange),ctx = ctx)
        self.logit2.initialize(init.Uniform(initrange),ctx = ctx)
        self.regress1.initialize(init.Uniform(initrange),ctx = ctx)
        self.regress2.initialize(init.Uniform(initrange),ctx = ctx)
        self.core.initialize(init.Uniform(initrange),ctx = ctx)
    def init_hidden(self, bsz,ctx):
        return (nd.zeros(shape=(self.num_layers, bsz, self.rnn_size),ctx = ctx),
                nd.zeros(shape = (self.num_layers, bsz, self.rnn_size),ctx = ctx))

    def forward(self, x, y, sample_prob=None):
        if sample_prob is not None:
            self.sample_prob = sample_prob
        batch_size = x.shape[0]
        state = self.init_hidden(batch_size,self.ctx)
        outputs_pgm = []
        outputs_param = []
        seq = y
        for i in range(seq.shape[1]):
            if i == 0:
                xt = x
            else:
                if i >= 1 and self.sample_prob > 0:
                    #print("x.shape:",x.shape)
                    sample_prob = nd.uniform(0,1,shape = (batch_size),ctx = self.ctx)  #sample_prob.shape (10,)
                    sample_mask = sample_prob < self.sample_prob
                    #print("sample_mask:",sample_mask)
                    #print("sample_mask.sum:",sample_mask.sum().asscalar())
                    if sample_mask.sum() == 0:
                        it1 = seq[:, i-1]
                    else:
                        sample_ind = sample_mask!=0
                        #print("sample_ind:",sample_ind)
                        it1 = seq[:, i-1]                         #it1.shape : (10,) 
                        #print("it1:",it1.shape)
                        #print("output_prog:",outputs_pgm[-1])
                        prob_prev = nd.exp(outputs_pgm[-1])
                        #print("prob_pre:",prob_prev)
                        temp= nd.random.multinomial(prob_prev,1).reshape(-1).astype('int64')
                        #print("prob_prev:",nd.argmax(prob_prev,axis=1).astype('int64')==temp)
                        #print("temp",temp,"\n it1:",it1)
                        it1 = nd.where(sample_ind,temp,it1).astype('float32')
                else:
                    #print("obtain last ground truth")
                    it1 = seq[:, i-1].copy()
                xt = self.pgm_embed(it1)
                #print("xt after embed:",xt)
                
            #print("xt                      :",xt)
            output, state = self.core(xt.expand_dims(axis = 0), state)

            pgm_feat1 = nd.relu(self.logit1(output.squeeze(0)))
            pgm_feat2 = self.logit2(pgm_feat1)
            pgm_score = nd.log_softmax(pgm_feat2, axis=1)

            trans_prob = nd.softmax(pgm_feat2, axis=1).detach()
            param_feat1 = nd.relu(self.regress1(output.squeeze(0)))
            param_feat2 = nd.concat(trans_prob, param_feat1, dim=1)
            
            param_score = self.regress2(param_feat2)
            param_score = param_score.reshape(batch_size, self.vocab_size + 1, self.max_param)
            #index = nd.argmax(trans_prob, axis = 1)
            index = seq[:,i]
            index = index.expand_dims(axis = 1).expand_dims(axis = 2).broadcast_to(shape=(batch_size, 1,self.max_param)).detach()
            param_score = nd.pick(param_score,index,1)
            
            outputs_pgm.append(pgm_score)
            outputs_param.append(param_score)
            
        outputs_pgm = [_.expand_dims(axis = 1) for _ in outputs_pgm]
        outputs_param = [_.expand_dims(axis = 1) for _ in outputs_param]
        pgms = outputs_pgm[0]
        params = outputs_param[0]
        for i in range(1,len(outputs_pgm)):
            pgms = nd.concat(pgms,outputs_pgm[i],dim = 1)
            params = nd.concat(params,outputs_param[i],dim = 1)
        #print("params", params.shape)
        #rint("pgm", pgms.shape)
            
        return [pgms,params]

    def decode(self, x):
        batch_size = x.shape[0]
        state = self.init_hidden(batch_size,self.ctx)
        outputs_pgm = []
        outputs_param = []
        
        for i in range(self.seq_length):
            if i == 0:
                xt = x
            else:
                prob_pre = nd.exp(outputs_pgm[-1])
                it1 = nd.argmax(prob_pre, axis=1)
                #print("it1 decode:",it1)
                xt = self.pgm_embed(it1)
            #print("xt decode:",xt)
            output, state = self.core(xt.expand_dims(axis=0), state)

            pgm_feat1 = nd.relu(self.logit1(output.squeeze(0)))
            pgm_feat2 = self.logit2(pgm_feat1)
            pgm_score = nd.log_softmax(pgm_feat2, axis=1)

            trans_prob = nd.softmax(pgm_feat2, axis=1).detach()
            param_feat1 = nd.relu(self.regress1(output.squeeze(0)))
            param_feat2 = nd.concat(trans_prob, param_feat1, dim=1)
            param_score = self.regress2(param_feat2)
            param_score = param_score.reshape(batch_size, self.vocab_size + 1, self.max_param)

            index = nd.argmax(trans_prob, axis = 1)
            index = index.expand_dims(axis = 1).expand_dims(axis = 2).broadcast_to(shape=(batch_size, 1,self.max_param)).detach() ##
            param_score = nd.pick(param_score,index,1)

            outputs_pgm.append(pgm_score)
            outputs_param.append(param_score)
        outputs_pgm = [_.expand_dims(axis = 1) for _ in outputs_pgm]
        outputs_param = [_.expand_dims(axis = 1) for _ in outputs_param]
        pgms = outputs_pgm[0]
        params = outputs_param[0]
        for i in range(1,len(outputs_pgm)):
            pgms = nd.concat(pgms,outputs_pgm[i],dim = 1)
            params = nd.concat(params,outputs_param[i],dim = 1)
        return [pgms,params]


class Conv3DNet(nn.Block):
    """
    encode 3D voxelized shape into a vector
    """
    def __init__(self, feat_size, input_channel=1, power=1,**kwargs):
        super(Conv3DNet, self).__init__(**kwargs)

        power = int(power)
        self.parameters = [(8,5,1,2),(16,3,2,1),(16,3,1,1),(32,3,2,1),(32,3,1,1),
                          (64,3,2,1),(64,3,1,1),(64,3,1,1)]
        self.block = nn.Sequential()
        for params in self.parameters:
            self.block.add(nn.Conv3D(channels = params[0]*power, kernel_size = params[1], 
                                     strides= params[2], padding= params[3]),
                          nn.BatchNorm(),
                          nn.Activation('relu'))
        
        self.avgpool = nn.AvgPool3D(pool_size=(4, 4, 4))

        self.fc = nn.Dense(feat_size)

    def forward(self, x):
        x = self.block(x);
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = nd.relu(self.fc(x))
        
        return x


class RenderNet(nn.Block):
    """
    Multiple Step Render
    """
    def __init__(self, opt):
        super(RenderNet, self).__init__()

        # program LSTM parameter
        self.vocab_size = opt.program_size
        self.max_param = opt.max_param
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.program_vector_size = opt.program_vector_size
        self.nc = opt.nc
        
        
        self.pgm_embed = nn.Dense(int(self.input_encoding_size / 2))
        self.param_embed = nn.Dense(self.input_encoding_size - int(self.input_encoding_size / 2))
        self.ctx = d2l.try_gpu()
        self.lstm = rnn.LSTM(hidden_size = self.rnn_size,
                                num_layers = self.num_layers,
                                dropout = self.drop_prob_lm,
                                input_size = self.input_encoding_size)
        
        self.pgm_param_feat = nn.Dense(self.program_vector_size)

        self.decoder = nn.Sequential()
        params = [(64,4,1,0),(64,3,1,1),(16,4,2,1),(16,3,1,1),(4, 4, 2, 1),(4, 3, 1, 1)]
        for idx,param in enumerate(params):
            if idx%2==0:
                self.decoder.add(nn.Conv3DTranspose(channels = param[0],kernel_size = param[1],
                                                    strides = param[2],padding = param[3],use_bias=False),
                                nn.BatchNorm(),
                                nn.Activation('relu'))
            else:
                self.decoder.add(nn.Conv3D(channels = param[0],kernel_size = param[1],
                                           strides = param[2],padding = param[3],use_bias = False),
                                nn.BatchNorm(),
                                nn.Activation('relu'))
                
        self.decoder.add(
            nn.Conv3DTranspose(self.nc, 4, 2, 1, use_bias=False))
    #def init_blocks(self,ctx):
        
    def init_hidden(self, bsz,ctx):
        return (nd.zeros(shape=(self.num_layers, bsz, self.rnn_size),ctx = ctx),
                nd.zeros(shape = (self.num_layers, bsz, self.rnn_size),ctx = ctx))

    def forward(self, program, parameters, index):
        program = program.transpose((1, 0, 2))
        parameters = parameters.transpose((1, 0, 2))
        bsz = program.shape[1]
        state = self.init_hidden(bsz,self.ctx)
        #print("program.shape:",program.shape)   (3, bsz, 22)
        #print("param.shape:",parameters.shape)  (3, bsz, 7)
        
        # program linear transform
        dim1 = program.shape
        program = program.reshape(-1, self.vocab_size + 1)
        x1 = nd.relu(self.pgm_embed(program))
        x1 = x1.reshape(dim1[0], dim1[1], -1)
        #print("program.shape after embeding:",x1.shape)   (3, bsz, 64)
        
        # parameter linear transform
        dim2 = parameters.shape
        parameters = parameters.reshape(-1, self.max_param)
        x2 = nd.relu(self.param_embed(parameters))
        x2 = x2.reshape(dim2[0], dim2[1], -1)
        #print("param.shape after embeding:",x2.shape)   (3, bsz, 64)
        
        # LSTM to aggregate programs and parameters
        x = nd.concat(x1, x2, dim=2)
        out, hidden = self.lstm(x, state)
        #print("lstm_out.shape:",out.shape)   (3, bsz, 128)
        #print("index.shape:",index.shape)    (bsz,)
        
        # select desired step aggregated features
        #print("index.shape:",index.shape,"out.shape:",out.shape)
        index = index.expand_dims(axis = 1).broadcast_to((bsz, out.shape[2])).expand_dims(axis = 0)
        #index = index.expand_dims(axis = 2).transpose((1,0,2)).broadcast_to((out.shape[0],bsz, out.shape[2]))
        #print("index.shape:",index,"\nout:",out.shape)   #(1,bsz,128)
        pgm_param_feat = nd.pick(out,index,0).squeeze()
        #print("pgm_param_feat.shape:",pgm_param_feat.shape)   #(bsz,128)
        pgm_param_feat = nd.relu(self.pgm_param_feat(pgm_param_feat))
        #print("pgm_param_feat.shape:",pgm_param_feat.shape)   (bsz,128)

        pgm_param_feat = pgm_param_feat.reshape(bsz, self.program_vector_size, 1, 1, 1)
        shape = self.decoder(pgm_param_feat)

        return shape

    def compute_LSTM_feat(self, program, parameters, index):
        program = program.permute(1, 0, 2)
        parameters = parameters.permute(1, 0, 2)
        bsz = program.x.shape[1]
        init = self.init_hidden(bsz)

        # program linear transform
        dim1 = program.x.shape
        program = program.reshape(-1, self.vocab_size + 1)
        x1 = nd.relu(self.pgm_embed(program))
        x1 = x1.reshape(dim1[0], dim1[1], -1)

        # parameter linear transform
        dim2 = parameters.x.shape
        parameters = parameters.reshape(-1, self.max_param)
        x2 = nd.relu(self.param_embed(parameters))
        x2 = x2.reshape(dim2[0], dim2[1], -1)

        # LSTM to aggregate programs and parameters
        x = nd.concat([x1, x2], axis=2)
        out, hidden = self.lstm(x, init)

        # select desired step aggregated features
        index = index.expand_dims(axis = 1).broadcast_to((-1, out.x.shape[2])).expand_dims(axis = 0)
        pgm_param_feat = gather(out,dim=0, index=index).squeeze()
        #pgm_param_feat = nd.relu(self.pgm_param_feat(pgm_param_feat))

        return pgm_param_feat

    def compute_shape_from_feat(self, pgm_param_feat):
        bsz = pgm_param_feat.x.shape[0]
        pgm_param_feat = nd.relu(self.pgm_param_feat(pgm_param_feat))
        pgm_param_feat = pgm_param_feat.reshape(bsz, self.program_vector_size, 1, 1, 1)
        shape = self.decoder(pgm_param_feat)

        return shape


if __name__ == '__main__':

    from easydict import EasyDict as edict
    from programs.label_config import stop_id, max_param

    opt = edict()
    opt.shape_feat_size = 64

    opt.outer_input_size = 64
    opt.outer_rnn_size = 64
    opt.outer_num_layers = 1
    opt.outer_drop_prob = 0
    opt.outer_seq_length = 6
    opt.is_cuda = False
    opt.program_size = stop_id - 1
    opt.max_param = max_param - 1
    opt.inner_input_size = 64
    opt.inner_rnn_size = 64
    opt.inner_num_layers = 1
    opt.inner_drop_prob = 0
    opt.inner_seq_length = 3
    opt.inner_cls_feat_size = 64
    opt.inner_reg_feat_size = 64
    opt.inner_sample_prob = 1.0

    net = BlockOuterNet(opt)
    ctx = d2l.try_gpu()
    net.init_blocks(ctx)
    x = nd.zeros((10, 1, 32, 32, 32)).as_in_context(ctx)
    y = nd.zeros((10, 6, 3)).as_in_context(ctx)
    pgms, params = net(x, y)
    print(pgms.shape)
    print(params.shape)
    pgms, params = net.decode(x)
    print(pgms.shape)
    print(params.shape)