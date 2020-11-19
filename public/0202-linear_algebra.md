# Linear Algebra with Torch {#linearalgebra}

_Last update: Thu Oct 22 16:46:28 2020 -0500 (54a46ea04)_

The following are basic operations of Linear Algebra using PyTorch.



```r
library(rTorch)
```


## Scalars


```r
torch$scalar_tensor(2.78654)

torch$scalar_tensor(0L)

torch$scalar_tensor(1L)

torch$scalar_tensor(TRUE)

torch$scalar_tensor(FALSE)
```

```
#> tensor(2.7865)
#> tensor(0.)
#> tensor(1.)
#> tensor(1.)
#> tensor(0.)
```

## Vectors


```r
v <- c(0, 1, 2, 3, 4, 5)
torch$as_tensor(v)
```

```
#> tensor([0., 1., 2., 3., 4., 5.])
```

### Vector to matrix


```r
# row-vector
message("R matrix")
```

```
#> R matrix
```

```r
(mr <- matrix(1:10, nrow=1))
message("as_tensor")
```

```
#> as_tensor
```

```r
torch$as_tensor(mr)
message("shape_of_tensor")
```

```
#> shape_of_tensor
```

```r
torch$as_tensor(mr)$shape
```

```
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    1    2    3    4    5    6    7    8    9    10
#> tensor([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]], dtype=torch.int32)
#> torch.Size([1, 10])
```

### Matrix to tensor


```r
# column-vector
message("R matrix, one column")
```

```
#> R matrix, one column
```

```r
(mc <- matrix(1:10, ncol=1))
message("as_tensor")
```

```
#> as_tensor
```

```r
torch$as_tensor(mc)
message("size of tensor")
```

```
#> size of tensor
```

```r
torch$as_tensor(mc)$shape
```

```
#>       [,1]
#>  [1,]    1
#>  [2,]    2
#>  [3,]    3
#>  [4,]    4
#>  [5,]    5
#>  [6,]    6
#>  [7,]    7
#>  [8,]    8
#>  [9,]    9
#> [10,]   10
#> tensor([[ 1],
#>         [ 2],
#>         [ 3],
#>         [ 4],
#>         [ 5],
#>         [ 6],
#>         [ 7],
#>         [ 8],
#>         [ 9],
#>         [10]], dtype=torch.int32)
#> torch.Size([10, 1])
```

## Matrices


```r
message("R matrix")
```

```
#> R matrix
```

```r
(m1 <- matrix(1:24, nrow = 3, byrow = TRUE))
message("as_tensor")
```

```
#> as_tensor
```

```r
(t1 <- torch$as_tensor(m1))
message("shape")
```

```
#> shape
```

```r
torch$as_tensor(m1)$shape
message("size")
```

```
#> size
```

```r
torch$as_tensor(m1)$size()
message("dim")
```

```
#> dim
```

```r
dim(torch$as_tensor(m1))
message("length")
```

```
#> length
```

```r
length(torch$as_tensor(m1))
```

```
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8]
#> [1,]    1    2    3    4    5    6    7    8
#> [2,]    9   10   11   12   13   14   15   16
#> [3,]   17   18   19   20   21   22   23   24
#> tensor([[ 1,  2,  3,  4,  5,  6,  7,  8],
#>         [ 9, 10, 11, 12, 13, 14, 15, 16],
#>         [17, 18, 19, 20, 21, 22, 23, 24]], dtype=torch.int32)
#> torch.Size([3, 8])
#> torch.Size([3, 8])
#> [1] 3 8
#> [1] 24
```


```r
message("R matrix")
```

```
#> R matrix
```

```r
(m2 <- matrix(0:99, ncol = 10))
message("as_tensor")
```

```
#> as_tensor
```

```r
(t2 <- torch$as_tensor(m2))
message("shape")
```

```
#> shape
```

```r
t2$shape
message("dim")
```

```
#> dim
```

```r
dim(torch$as_tensor(m2))
```

```
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,]    0   10   20   30   40   50   60   70   80    90
#>  [2,]    1   11   21   31   41   51   61   71   81    91
#>  [3,]    2   12   22   32   42   52   62   72   82    92
#>  [4,]    3   13   23   33   43   53   63   73   83    93
#>  [5,]    4   14   24   34   44   54   64   74   84    94
#>  [6,]    5   15   25   35   45   55   65   75   85    95
#>  [7,]    6   16   26   36   46   56   66   76   86    96
#>  [8,]    7   17   27   37   47   57   67   77   87    97
#>  [9,]    8   18   28   38   48   58   68   78   88    98
#> [10,]    9   19   29   39   49   59   69   79   89    99
#> tensor([[ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#>         [ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
#>         [ 2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
#>         [ 3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
#>         [ 4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
#>         [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
#>         [ 6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
#>         [ 7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
#>         [ 8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
#>         [ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]], dtype=torch.int32)
#> torch.Size([10, 10])
#> [1] 10 10
```


```r
m1[1, 1]
m2[1, 1]
```

```
#> [1] 1
#> [1] 0
```


```r
t1[1, 1]
t2[1, 1]
```

```
#> tensor(1, dtype=torch.int32)
#> tensor(0, dtype=torch.int32)
```

## 3D+ tensors


```r
# RGB color image has three axes 
(img <- torch$rand(3L, 28L, 28L))
img$shape
```

```
#> tensor([[[0.4349, 0.1164, 0.5637,  ..., 0.7674, 0.0530, 0.5104],
#>          [0.5074, 0.0026, 0.8199,  ..., 0.1035, 0.9890, 0.0948],
#>          [0.5082, 0.6629, 0.4485,  ..., 0.2037, 0.5876, 0.7726],
#>          ...,
#>          [0.9531, 0.4397, 0.1301,  ..., 0.9004, 0.7199, 0.6334],
#>          [0.2234, 0.0349, 0.3215,  ..., 0.9437, 0.9297, 0.9696],
#>          [0.5090, 0.7271, 0.0736,  ..., 0.3271, 0.0580, 0.7623]],
#> 
#>         [[0.0232, 0.7732, 0.9972,  ..., 0.4132, 0.1901, 0.6690],
#>          [0.3026, 0.6929, 0.1662,  ..., 0.8764, 0.8435, 0.3876],
#>          [0.6784, 0.5015, 0.4514,  ..., 0.9874, 0.0386, 0.1774],
#>          ...,
#>          [0.3697, 0.0044, 0.4686,  ..., 0.9114, 0.5276, 0.0438],
#>          [0.3210, 0.0769, 0.4184,  ..., 0.1150, 0.0206, 0.3720],
#>          [0.6467, 0.1786, 0.5240,  ..., 0.2346, 0.0390, 0.2670]],
#> 
#>         [[0.9525, 0.0805, 0.0763,  ..., 0.5606, 0.2202, 0.5187],
#>          [0.0708, 0.3832, 0.7780,  ..., 0.6198, 0.0404, 0.4178],
#>          [0.8492, 0.3753, 0.2217,  ..., 0.4277, 0.1597, 0.9825],
#>          ...,
#>          [0.0025, 0.2161, 0.5639,  ..., 0.8237, 0.4728, 0.0648],
#>          [0.8162, 0.7106, 0.0972,  ..., 0.4748, 0.0605, 0.7730],
#>          [0.8349, 0.5473, 0.5700,  ..., 0.7152, 0.1603, 0.5442]]])
#> torch.Size([3, 28, 28])
```


```r
img[1, 1, 1]
img[3, 28, 28]
```

```
#> tensor(0.4349)
#> tensor(0.5442)
```


## Transpose of a matrix


```r
(m3 <- matrix(1:25, ncol = 5))

# transpose
message("transpose")
```

```
#> transpose
```

```r
tm3 <- t(m3)
tm3
```

```
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    6   11   16   21
#> [2,]    2    7   12   17   22
#> [3,]    3    8   13   18   23
#> [4,]    4    9   14   19   24
#> [5,]    5   10   15   20   25
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,]    1    2    3    4    5
#> [2,]    6    7    8    9   10
#> [3,]   11   12   13   14   15
#> [4,]   16   17   18   19   20
#> [5,]   21   22   23   24   25
```


```r
message("as_tensor")
```

```
#> as_tensor
```

```r
(t3 <- torch$as_tensor(m3))
message("transpose")
```

```
#> transpose
```

```r
tt3 <- t3$transpose(dim0 = 0L, dim1 = 1L)
tt3
```

```
#> tensor([[ 1,  6, 11, 16, 21],
#>         [ 2,  7, 12, 17, 22],
#>         [ 3,  8, 13, 18, 23],
#>         [ 4,  9, 14, 19, 24],
#>         [ 5, 10, 15, 20, 25]], dtype=torch.int32)
#> tensor([[ 1,  2,  3,  4,  5],
#>         [ 6,  7,  8,  9, 10],
#>         [11, 12, 13, 14, 15],
#>         [16, 17, 18, 19, 20],
#>         [21, 22, 23, 24, 25]], dtype=torch.int32)
```


```r
tm3 == tt3$numpy()   # convert first the tensor to numpy
```

```
#>      [,1] [,2] [,3] [,4] [,5]
#> [1,] TRUE TRUE TRUE TRUE TRUE
#> [2,] TRUE TRUE TRUE TRUE TRUE
#> [3,] TRUE TRUE TRUE TRUE TRUE
#> [4,] TRUE TRUE TRUE TRUE TRUE
#> [5,] TRUE TRUE TRUE TRUE TRUE
```

## Vectors, special case of a matrix


```r
message("R matrix")
```

```
#> R matrix
```

```r
m2 <- matrix(0:99, ncol = 10)
message("as_tensor")
```

```
#> as_tensor
```

```r
(t2 <- torch$as_tensor(m2))

# in R
message("select column of matrix")
```

```
#> select column of matrix
```

```r
(v1 <- m2[, 1])
message("select row of matrix")
```

```
#> select row of matrix
```

```r
(v2 <- m2[10, ])
```

```
#> tensor([[ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#>         [ 1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
#>         [ 2, 12, 22, 32, 42, 52, 62, 72, 82, 92],
#>         [ 3, 13, 23, 33, 43, 53, 63, 73, 83, 93],
#>         [ 4, 14, 24, 34, 44, 54, 64, 74, 84, 94],
#>         [ 5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
#>         [ 6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
#>         [ 7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
#>         [ 8, 18, 28, 38, 48, 58, 68, 78, 88, 98],
#>         [ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]], dtype=torch.int32)
#>  [1] 0 1 2 3 4 5 6 7 8 9
#>  [1]  9 19 29 39 49 59 69 79 89 99
```


```r
# PyTorch
message()
```

```
#> 
```

```r
t2c <- t2[, 1]
t2r <- t2[10, ]

t2c
t2r
```

```
#> tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)
#> tensor([ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], dtype=torch.int32)
```

In vectors, the vector and its transpose are equal.


```r
tt2r <- t2r$transpose(dim0 = 0L, dim1 = 0L)
tt2r
```

```
#> tensor([ 9, 19, 29, 39, 49, 59, 69, 79, 89, 99], dtype=torch.int32)
```


```r
# a tensor of booleans. is vector equal to its transposed?
t2r == tt2r
```

```
#> tensor([True, True, True, True, True, True, True, True, True, True])
```

## Tensor arithmetic


```r
message("x")
```

```
#> x
```

```r
(x = torch$ones(5L, 4L))
message("y")
```

```
#> y
```

```r
(y = torch$ones(5L, 4L))
message("x+y")
```

```
#> x+y
```

```r
x + y
```

```
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
#> tensor([[2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.],
#>         [2., 2., 2., 2.]])
```

$$A + B = B + A$$


```r
x + y == y + x
```

```
#> tensor([[True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True]])
```

## Add a scalar to a tensor


```r
s <- 0.5    # scalar
x + s
```

```
#> tensor([[1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000],
#>         [1.5000, 1.5000, 1.5000, 1.5000]])
```


```r
# scalar multiplying two tensors
s * (x + y)
```

```
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
```

## Multiplying tensors

$$A * B = B * A$$


```r
message("x")
```

```
#> x
```

```r
(x = torch$ones(5L, 4L))
message("y")
```

```
#> y
```

```r
(y = torch$ones(5L, 4L))
message("2x+4y")
```

```
#> 2x+4y
```

```r
(z = 2 * x + 4 * y)
```

```
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
#> tensor([[1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.],
#>         [1., 1., 1., 1.]])
#> tensor([[6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.],
#>         [6., 6., 6., 6.]])
```



```r
x * y == y * x
```

```
#> tensor([[True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True],
#>         [True, True, True, True]])
```



## Dot product

$$dot(a,b)_{i,j,k,a,b,c} = \sum_m a_{i,j,k,m}b_{a,b,m,c}$$


```r
torch$dot(torch$tensor(c(2, 3)), torch$tensor(c(2, 1)))
```

```
#> tensor(7.)
```

### 2D array using Python


```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[1, 2], [3, 4]])
print(a)
```

```
#> [[1 2]
#>  [3 4]]
```

```python
print(b)
```

```
#> [[1 2]
#>  [3 4]]
```

```python
np.dot(a, b)
```

```
#> array([[ 7, 10],
#>        [15, 22]])
```

### 2D array using R


```r
a <- np$array(list(list(1, 2), list(3, 4)))
a
b <- np$array(list(list(1, 2), list(3, 4)))
b

np$dot(a, b)
```

```
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4
#>      [,1] [,2]
#> [1,]    1    2
#> [2,]    3    4
#>      [,1] [,2]
#> [1,]    7   10
#> [2,]   15   22
```

`torch.dot()` treats both $a$ and $b$ as __1D__ vectors (irrespective of their original shape) and computes their inner product. 


```r
at <- torch$as_tensor(a)
bt <- torch$as_tensor(b)

# torch$dot(at, bt)  <- RuntimeError: dot: Expected 1-D argument self, but got 2-D
# at %.*% bt
```

If we perform the same dot product operation in Python, we get the same error:


```python
import torch
import numpy as np

a = np.array([[1, 2], [3, 4]])
a
```

```
#> array([[1, 2],
#>        [3, 4]])
```

```python
b = np.array([[1, 2], [3, 4]])
b
```

```
#> array([[1, 2],
#>        [3, 4]])
```

```python
np.dot(a, b)
```

```
#> array([[ 7, 10],
#>        [15, 22]])
```

```python
at = torch.as_tensor(a)
bt = torch.as_tensor(b)

at
```

```
#> tensor([[1, 2],
#>         [3, 4]])
```

```python
bt
```

```
#> tensor([[1, 2],
#>         [3, 4]])
```

```python
torch.dot(at, bt)
```

```
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: 1D tensors expected, got 2D, 2D tensors at /opt/conda/conda-bld/pytorch_1595629401553/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:83
#> 
#> Detailed traceback: 
#>   File "<string>", line 1, in <module>
```



```r
a <- torch$Tensor(list(list(1, 2), list(3, 4)))
b <- torch$Tensor(c(c(1, 2), c(3, 4)))
c <- torch$Tensor(list(list(11, 12), list(13, 14)))

a
b
torch$dot(a, b)
```

```
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: 1D tensors expected, got 2D, 1D tensors at /opt/conda/conda-bld/pytorch_1595629401553/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:83
```

```r
# this is another way of performing dot product in PyTorch
# a$dot(a)
```

```
#> tensor([[1., 2.],
#>         [3., 4.]])
#> tensor([1., 2., 3., 4.])
```


```r
o1 <- torch$ones(2L, 2L)
o2 <- torch$ones(2L, 2L)

o1
o2

torch$dot(o1, o2)
```

```
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: 1D tensors expected, got 2D, 2D tensors at /opt/conda/conda-bld/pytorch_1595629401553/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:83
```

```r
o1$dot(o2)
```

```
#> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: 1D tensors expected, got 2D, 2D tensors at /opt/conda/conda-bld/pytorch_1595629401553/work/aten/src/TH/generic/THTensorEvenMoreMath.cpp:83
```

```
#> tensor([[1., 1.],
#>         [1., 1.]])
#> tensor([[1., 1.],
#>         [1., 1.]])
```



```r
# 1D tensors work fine
r = torch$dot(torch$Tensor(list(4L, 2L, 4L)), torch$Tensor(list(3L, 4L, 1L)))
r
```

```
#> tensor(24.)
```

### `mm` and `matmul` functions
So, if we cannor perform 2D tensor operations with the `dot` product, how do we manage then?


```r
## mm and matmul seem to address the dot product we are looking for in tensors
a = torch$randn(2L, 3L)
b = torch$randn(3L, 4L)

a$mm(b)
a$matmul(b)
```

```
#> tensor([[ 1.0735,  2.0763, -0.2199,  0.3611],
#>         [-1.3501,  4.1254, -2.2058,  0.8386]])
#> tensor([[ 1.0735,  2.0763, -0.2199,  0.3611],
#>         [-1.3501,  4.1254, -2.2058,  0.8386]])
```

Here is a good explanation: https://stackoverflow.com/a/44525687/5270873

Let's now prove the associative property of tensors:

$$(A B)^T = B^T A^T$$


```r
abt <- torch$mm(a, b)$transpose(dim0=0L, dim1=1L)
abt
```

```
#> tensor([[ 1.0735, -1.3501],
#>         [ 2.0763,  4.1254],
#>         [-0.2199, -2.2058],
#>         [ 0.3611,  0.8386]])
```


```r
at <- a$transpose(dim0=0L, dim1=1L)
bt <- b$transpose(dim0=0L, dim1=1L)

btat <- torch$matmul(bt, at)
btat
```

```
#> tensor([[ 1.0735, -1.3501],
#>         [ 2.0763,  4.1254],
#>         [-0.2199, -2.2058],
#>         [ 0.3611,  0.8386]])
```

And we could unit test if the results are nearly the same with `allclose()`:


```r
# tolerance
torch$allclose(abt, btat, rtol=0.0001)
```

```
#> [1] TRUE
```

