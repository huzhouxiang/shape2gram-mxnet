from __future__ import print_function

import os
import h5py
import numpy as np
from programs.sample_blocks import sample_batch



def get_data(n_samples,path,ratio):
    n_samples = [int(a*ratio) for a in n_samples]
    num_samples = sum(n_samples)
    f_train = h5py.File(path, 'w')
    f_train.create_dataset("data",(n_samples[0],32,32,32),maxshape = (None,32,32,32),dtype = np.uint8)
    f_train.create_dataset("label",(n_samples[0],3,8),maxshape = (None,3,8), dtype = np.int32)
    f_train.close()
    n = 0
    for i in range(len(n_samples)):
        with h5py.File(path,'a') as f_add:
            train_data = f_add['data']
            label = f_add['label']
            #resize arrays to store additional data
            train_data.resize((n+n_samples[i],32,32,32))
            label.resize((n+n_samples[i],3,8))
            d, s = sample_batch(num=n_samples[i], primitive_type=i)
            train_data[n:n+n_samples[i]] = d
            label[n:n+n_samples[i]] = s
            n+=n_samples[i]
    print('==> saving %s data total %d'%(path[7:-3],n))
    
train_file = './data/train_blocks.h5'
val_file = './data/val_blocks.h5'   
n_samples_train = [5000,
             30000, 5000, 5000, 5000, 10000,
             5000, 5000, 5000, 5000, 30000,
             15000, 30000, 8000, 5000, 30000,
             15000, 6000, 6000, 6000, 10000,
             30000, 10000, 6000, 6000, 6000,
             30000, 40000, 30000, 10000]


n_samples_val = [30000, 30000, 40000, 30000, 30000,
             30000, 20000, 10000, 40000, 35000,
             40000, 35000, 35000, 35000, 50000]



get_data(n_samples_train,train_file,0.5)

get_data(n_samples_val,val_file,0.5)

print('Done')
