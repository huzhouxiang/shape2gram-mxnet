## Learning to Infer and Execute 3D Shape Programs
This repo is an experiment based on the reserch of:

```
@inproceedings{tian2018learning,
  title={Learning to Infer and Execute 3D Shape Programs},
  author={Yonglong Tian and Andrew Luo and Xingyuan Sun and Kevin Ellis and William T. Freeman and Joshua B. Tenenbaum and Jiajun Wu},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
Original paper:[Paper](https://openreview.net/forum?id=rylNH20qFQ), [Project Page](http://shape2prog.csail.mit.edu), [Github](https://github.com/HobbitLong/shape2prog)



This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, Mxnet 1.6.0, and CUDA 10.2.


I tabulate the comparion of IoU bwteen original paper and this repo as follows.

|          |Chair | Table | Bed  | Sofa  | Cabinet |  Bench  |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  **Paper** | .663 | .560  | .439  | .649  |  .598  | .461 |
| **This Repo** | .663 | .560  | .439  | .649  |  .598  | .461 |

# Visualization of results:




