from __future__ import print_function

import mxnet as mx
from mxnet import nd
import numpy as np
from d2l import mxnet as d2l

from programs.utils import draw_vertical_leg as draw_vertical_leg_new
from programs.utils import draw_rectangle_top as draw_rectangle_top_new
from programs.utils import draw_square_top as draw_square_top_new
from programs.utils import draw_circle_top as draw_circle_top_new
from programs.utils import draw_middle_rect_layer as draw_middle_rect_layer_new
from programs.utils import draw_circle_support as draw_circle_support_new
from programs.utils import draw_square_support as draw_square_support_new
from programs.utils import draw_circle_base as draw_circle_base_new
from programs.utils import draw_square_base as draw_square_base_new
from programs.utils import draw_cross_base as draw_cross_base_new
from programs.utils import draw_sideboard as draw_sideboard_new
from programs.utils import draw_horizontal_bar as draw_horizontal_bar_new
from programs.utils import draw_vertboard as draw_vertboard_new
from programs.utils import draw_locker as draw_locker_new
from programs.utils import draw_tilt_back as draw_tilt_back_new
from programs.utils import draw_chair_beam as draw_chair_beam_new
from programs.utils import draw_line as draw_line_new
from programs.utils import draw_back_support as draw_back_support_new

from programs.loop_gen import decode_loop, translate, rotate, end


def get_distance_to_center():
    x = np.arange(32)
    y = np.arange(32)
    xx, yy = np.meshgrid(x, y)
    xx = xx + 0.5
    yy = yy + 0.5
    d = np.sqrt(np.square(xx - int(32 / 2)) + np.square(yy - int(32 / 2)))
    return d

def gather(self, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
        index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = nd.swapaxes(self, 0, dim).asnumpy()
    index_swaped = nd.swapaxes(index, 0, dim).asnumpy()
    #print(data_swaped,index_swaped)
    #print("index_swaped\n",index_swaped,index_swaped.shape,"data_swaped\n",data_swaped,data_swaped.shape,'\n')
    gathered = nd.from_numpy(np.choose(index_swaped,data_swaped)).as_in_context(d2l.try_gpu())
    return nd.swapaxes(gathered, 0, dim)

def scatter_numpy(self, dim, index, src):
    """
    Writes all values from the Tensor src into self at the indices specified in the index Tensor.

    :param dim: The axis along which to index
    :param index: The indices of elements to scatter
    :param src: The source element(s) to scatter
    :return: self
    """
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    if self.ndim != index.ndim:
        raise ValueError("Index should have the same number of dimensions as output")
    if dim >= self.ndim or dim < -self.ndim:
        raise IndexError("dim is out of range")
    if dim < 0:
        # Not sure why scatter should accept dim < 0, but that is the behavior in PyTorch's scatter
        dim = self.ndim + dim
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and output should be the same size")
    if (index >= self.shape[dim]).any() or (index < 0).any():
        raise IndexError("The values of index must be between 0 and (self.shape[dim] -1)")

    def make_slice(arr, dim, i):
        slc = [slice(None)] * arr.ndim
        slc[dim] = i
        return slc

    # We use index and dim parameters to create idx
    # idx is in a form that can be used as a NumPy advanced index for scattering of src param. in self
    idx = [[*np.indices(idx_xsection_shape).reshape(index.ndim - 1, -1),
            index[make_slice(index, dim, i)].reshape(1, -1)[0]] for i in range(index.shape[dim])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(dim, idx.pop())

    if not np.isscalar(src):
        if index.shape[dim] > src.shape[dim]:
            raise IndexError("Dimension " + str(dim) + "of index can not be bigger than that of src ")
        src_xsection_shape = src.shape[:dim] + src.shape[dim + 1:]
        if idx_xsection_shape != src_xsection_shape:
            raise ValueError("Except for dimension " +
                             str(dim) + ", all dimensions of index and src should be the same size")
        # src_idx is a NumPy advanced index for indexing of elements in the src
        src_idx = list(idx)
        src_idx.pop(dim)
        src_idx.insert(dim, np.repeat(np.arange(index.shape[dim]), np.prod(idx_xsection_shape)))
        self[idx] = src[src_idx]

    else:
        self[idx] = src

    return self



def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)





def get_last_block(pgm):
    bsz = pgm.size(0)
    n_block = pgm.size(1)
    n_step = pgm.size(2)
    pgm = pgm.copy() # not sure

    if pgm.dim() == 4:
        idx = nd.argmax(pgm, axis=3)
        idx = idx.as_in_context(mx.cpu()) # not sure
    elif pgm.dim() == 3:
        idx = pgm.as_in_context(mx.cpu())
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []
    for i in range(bsz):
        j = n_block - 1
        while j >= 0:
            if idx[i, j, 0] == 0:
                break
            j = j - 1

        if j == -1:
            max_inds.append(0)
        else:
            max_inds.append(j)

    return np.asarray(max_inds)


def sample_block(max_inds, include_tail=False):
    sample_inds = []
    for ind in max_inds:
        if include_tail:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def get_max_step_pgm(pgm):
    batch_size = pgm.size(0)
    pgm = pgm.copy() # not sure

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).as_in_context(mx.cpu())
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].as_in_context(mx.cpu())
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    max_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == 0:
            raise ValueError("no programs for such sample")
        max_inds.append(j)

    return np.asarray(max_inds)


def get_vacancy(pgm):
    batch_size = pgm.size(0)

    pgm = pgm.copy() # not sure

    if pgm.dim() == 3:
        pgm = pgm[:, 1:, :]
        idx = get_class(pgm).as_in_context(mx.cpu())
    elif pgm.dim() == 2:
        idx = pgm[:, 1:].as_in_context(mx.cpu())
    else:
        raise ValueError("pgm.dim() != 2 or 3")

    vac_inds = []

    for i in range(batch_size):
        j = 0
        while j < idx.shape[1]:
            if idx[i, j] == 0:
                break
            j = j + 1
        if j == idx.shape[1]:
            j = j - 1
        vac_inds.append(j)

    return np.asarray(vac_inds)


def sample_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(np.random.randint(0, ind + 1))
        else:
            sample_inds.append(np.random.randint(0, ind))
    return np.asarray(sample_inds)


def sample_last_ind(max_inds, include_start=False):
    sample_inds = []
    for ind in max_inds:
        if include_start:
            sample_inds.append(ind)
        else:
            sample_inds.append(ind - 1)
    return np.array(sample_inds)

def get_class(pgm):
    print(pgm)
    if len(pgm.shape) == 3:
        idx = nd.argmax(pgm, axis=2)
    elif len(pgm.shape) == 2:
        idx = pgm
    else:
        raise IndexError("dimension of pgm is wrong")
    return idx

def decode_to_shape_new(pred_pgm, pred_param):
    batch_size = pred_pgm.shape[0]

    idx = get_class(pred_pgm)

    pgm = idx.as_in_context(mx.cpu()).asnumpy()
    params = pred_param.as_in_context(mx.cpu()).asnumpy()
    params = np.round(params).astype(np.int32)

    data = np.zeros((batch_size, 32, 32, 32), dtype=np.uint8)
    for i in range(batch_size):
        for j in range(1, pgm.shape[1]):
            if pgm[i, j] == 0:
                continue
            data[i] = render_one_step_new(data[i], pgm[i, j], params[i, j])

    return data


def decode_pgm(pgm, param, loop_free=True):
    """
    decode and check one single block
    remove occasionally-happened illegal programs
    """
    flag = 1
    data_loop = []
    if pgm[0] == translate:
        if pgm[1] == translate:
            if 1 <= pgm[2] < translate:
                data_loop.append(np.hstack((pgm[0], param[0])))
                data_loop.append(np.hstack((pgm[1], param[1])))
                data_loop.append(np.hstack((pgm[2], param[2])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
                data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
            else:
                flag = 0
        elif 1 <= pgm[1] < translate:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif pgm[0] == rotate:
        if pgm[1] == 10:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        if pgm[1] == 17:
            data_loop.append(np.hstack((pgm[0], param[0])))
            data_loop.append(np.hstack((pgm[1], param[1])))
            data_loop.append(np.hstack(np.asarray([end, 0, 0, 0, 0, 0, 0, 0])))
        else:
            flag = 0
    elif 1 <= pgm[0] < translate:
        data_loop.append(np.hstack((pgm[0], param[0])))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    else:
        flag = 0

    if flag == 0:
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
        data_loop.append(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))

    data_loop = [x.tolist() for x in data_loop]
    data_loop_free = decode_loop(data_loop)
    data_loop_free = np.asarray(data_loop_free)

    if len(data_loop_free) == 0:
        data_loop_free = np.zeros((2, 8), dtype=np.int32)

    if loop_free:
        return data_loop_free
    else:
        return np.asarray(data_loop)


def decode_all(pgm, param, loop_free=False):
    """
    decode program to loop-free (or include loop)
    """
    n_block = pgm.shape[0]
    param = np.round(param).astype(np.int32)

    result = []
    for i in range(n_block):
        res = decode_pgm(pgm[i], param[i], loop_free=loop_free)
        result.append(res)
    result = np.concatenate(result, axis=0)
    return result


def execute_shape_program(pgm, param):
    """
    execute a single shape program
    """
    trace_sets = decode_all(pgm, param, loop_free=True)
    data = np.zeros((32, 32, 32), dtype=np.uint8)

    for trace in trace_sets:
        cur_pgm = trace[0]
        cur_param = trace[1:]
        data = render_one_step_new(data, cur_pgm, cur_param)

    return data


def decode_multiple_block(pgm, param):
    """
    decode and execute multiple blocks
    can run with batch style
    """
    # pgm: bsz x n_block x n_step x n_class
    # param: bsz x n_block x n_step x n_class
    bsz = pgm.shape[0]
    n_block = pgm.shape[1]
    data = np.zeros((bsz, 32, 32, 32), dtype=np.uint8)
    for i in range(n_block):
        if len(pgm.shape) == 4:
            prob_pre = nd.exp(pgm[:, i, :, :])
            it1 = nd.argmax(prob_pre, axis=2)
        elif len(pgm.shape) == 3:
            it1 = pgm[:, i, :]
        else:
            raise NotImplementedError('pgm has incorrect dimension')
        it2 = param[:, i, :, :].copy()
        it1 = it1.as_in_context(mx.cpu()).asnumpy()
        it2 = it2.as_in_context(mx.cpu()).asnumpy()
        data = render_block(data, it1, it2)

    return data


def count_blocks(pgm):
    """
    count the number of effective blocks
    """
    # pgm: bsz x n_block x n_step x n_class
    pgm = pgm.data.copy().as_in_context(mx.cpu())
    bsz = pgm.size(0)
    n_blocks = []
    n_for = []
    for i in range(bsz):
        prob = nd.exp(pgm[i, :, :, :])
        it = nd.argmax(prob, axis=2)
        v = it[:, 0].numpy()
        n_blocks.append((v > 0).sum())
        n_for.append((v == translate).sum() + (v == rotate).sum())

    return np.asarray(n_blocks), np.asarray(n_for)


def render_new(data, pgms, params):
    """
    render one step for a batch
    """
    batch_size = data.shape[0]
    params = np.round(params).astype(np.int32)

    for i in range(batch_size):
        data[i] = render_one_step_new(data[i], pgms[i], params[i])

    return data


def render_block(data, pgm, param):
    """
    render one single block
    """
    param = np.round(param).astype(np.int32)
    bsz = data.shape[0]
    for i in range(bsz):
        loop_free = decode_pgm(pgm[i], param[i])
        cur_pgm = loop_free[:, 0]
        cur_param = loop_free[:, 1:]
        for j in range(len(cur_pgm)):
            data[i] = render_one_step_new(data[i], cur_pgm[j], cur_param[j])

    return data


def render_one_step_new(data, pgm, param):
    """
    render one step
    """
    if pgm == 0:
        pass
    elif pgm == 1:
        data = draw_vertical_leg_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 2:
        data = draw_rectangle_top_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 3:
        data = draw_square_top_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 4:
        data = draw_circle_top_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 5:
        data = draw_middle_rect_layer_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 6:
        data = draw_circle_support_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 7:
        data = draw_square_support_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 8:
        data = draw_circle_base_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 9:
        data = draw_square_base_new(data, param[0], param[1], param[2], param[3], param[4])[0]
    elif pgm == 10:
        data = draw_cross_base_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 11:
        data = draw_sideboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 12:
        data = draw_horizontal_bar_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 13:
        data = draw_vertboard_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 14:
        data = draw_locker_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 15:
        data = draw_tilt_back_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6])[0]
    elif pgm == 16:
        data = draw_chair_beam_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    elif pgm == 17:
        data = draw_line_new(data, param[0], param[1], param[2], param[3], param[4], param[5], param[6])[0]
    elif pgm == 18:
        data = draw_back_support_new(data, param[0], param[1], param[2], param[3], param[4], param[5])[0]
    else:
        raise RuntimeError("program id is out of range, pgm={}".format(pgm))

    return data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
