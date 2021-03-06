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



|          | W/O GA | GA | GT  |
|----------|:----:|:---:|:---:|
|Chair| ![Chair(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/chair/images/Before%20GA%201.png) | ![Chair(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/chair/images/GA%201.png)  | ![Chair(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/chair/images/GT%201.png)|
|Table|![Table(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/table/images/Before%20GA%201.png)| ![Table(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/table/images/GA%201.png) | ![Table(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/table/images/GT%201.png)|
|Sofa| ![Sofa(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/sofa/images/Before%20GA%201.png)|![Sofa(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/sofa/images/GA%201.png)|![Sofa(GT):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/sofa/images/GT%201.png)|
|Cabinet|![Cabinet(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/cabinet/images/Before%20GA%201.png) | ![Cabinet(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/cabinet/images/GA%201.png)|![Cabinet(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/cabinet/images/GT%201.png)|
|Bed|![Bed(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bed/images/Before%20GA%201.png)|![Bed(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bed/images/GA%201.png)|![Bed(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bed/images/GT%201.png)|
|Bench|![Bench(befor GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bench/images/Before%20GA%201.png)|![Bench(after GA):](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bench/images/GA%201.png)|![Bench(GT)](https://github.com/huzhouxiang/shape2gram-mxnet/blob/main/output/bench/images/GT%201.png)|
