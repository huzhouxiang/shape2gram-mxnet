## Learning to Infer and Execute 3D Shape Programs
This repo is an experiment based on the research of:

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
|  **W/O GA** | 0.492 | 0.456  | 0.314  | 0.371  |  0.327 | 0.201 |
| **W GA** | 0.552 | **0.627**  | 0.405  | 0.629  |  0.549  | **0.464** |
| **Ori** | **0.663** | 0.560 | **0.439**  | **0.649**  |  **0.598**  | 0.461 |

where
"W/O GA": Without guided adaption;
"GA": After guided adaption;
"Ori": Original paper;


## Visualization of results:


Chair(befor GA):
![Chair(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/chair/images/Before%20GA%201.png)
Chair(after GA):
![Chair(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/chair/images/GA%201.png)
Chair(GT):
![Chair(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/chair/images/GT%201.png)

Table(befor GA):
![Table(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/table/images/Before%20GA%201.png)
Chair(after GA):
![Table(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/table/images/GA%201.png)
Chair(GT):
![Table(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/table/images/GT%201.png)

Sofa(befor GA):
![Sofa(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/sofa/images/Before%20GA%201.png)
Sofa(after GA):
![Sofa(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/sofa/images/GA%201.png)
Sofa(GT):
![Sofa(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/sofa/images/GT%201.png)


Cabinet(befor GA):
![Cabinet(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/cabinet/images/Before%20GA%201.png)
Cabinet(after GA):
![Cabinet(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/cabinet/images/GA%201.png)
Cabinet(GT):
![Cabinet(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/cabinet/images/GT%201.png)

Bed(befor GA):
![Bed(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bed/images/Before%20GA%201.png)
Bed(after GA):
![Bed(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bed/images/GA%201.png)
Bed(GT):
![Bed(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bed/images/GT%201.png)


Bench(befor GA):
![Bench(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bench/images/Before%20GA%201.png)
Bench(after GA):
![Bench(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bench/images/GA%201.png)
Bed(GT):
![Bench(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/tree/main/output/bench/images/GT%201.png)


