## Learning to Infer and Execute 3D Shape Programs


"Learning to Infer and Execute 3D Shape Programs" [Paper](https://openreview.net/forum?id=rylNH20qFQ), [Project Page](http://shape2prog.csail.mit.edu). 

![Teaser Image](http://shape2prog.csail.mit.edu/shape_files/teaser.jpg)



This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, Mxnet 1.6.0, and CUDA 10.2.


We tabulate the comparion of IoU bwteen original paper and this repo as follows.

|          |Chair | Table | Bed  | Sofa  | Cabinet |  Bench  |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  **Paper** | .591 | .516  | .367  | .597  |  .478  | .418 |
| **This Repo** | .663 | .560  | .439  | .649  |  .598  | .461 |


## Citation

This repo is based on the reserch of:

```
@inproceedings{tian2018learning,
  title={Learning to Infer and Execute 3D Shape Programs},
  author={Yonglong Tian and Andrew Luo and Xingyuan Sun and Kevin Ellis and William T. Freeman and Joshua B. Tenenbaum and Jiajun Wu},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```


