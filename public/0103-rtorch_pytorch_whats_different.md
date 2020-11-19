# rTorch vs PyTorch

*Last update: Sun Oct 25 13:00:41 2020 -0500 (265c0b3c1)*

## What's different

This chapter will explain the main differences between `PyTorch` and `rTorch`. Most of the things work directly in `PyTorch` but we need to be aware of some minor differences when working with rTorch. Here is a review of existing methods.

Let's start by loading `rTorch`:


```r
library(rTorch)
```

## Calling objects from PyTorch

We use the dollar sign or `$` to call a class, function or method from the `rTorch` modules. In this case, from the `torch` module:


```r
torch$tensor(c(1, 2, 3))
```

```
#> tensor([1., 2., 3.])
```

In Python, what we do is using the **dot** to separate the sub-members of an object:


```python
import torch
torch.tensor([1, 2, 3])
```

```
#> tensor([1, 2, 3])
```

## Call functions from `torch`


```r
library(rTorch)
# these are the equivalents of the Python import module
nn          <- torch$nn
transforms  <- torchvision$transforms
dsets       <- torchvision$datasets

torch$tensor(c(1, 2, 3))
```

```
#> tensor([1., 2., 3.])
```

The code above is equivalent to writing this code in Python:


```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

torch.tensor([1, 2, 3])
```

```
#> tensor([1, 2, 3])
```

Then we can proceed to extract classes, methods and functions from the `nn`, `transforms`, and `dsets` objects. In this example we use the module `torchvision$datasets` and the function `transforms$ToTensor()`. For example, the `train_dataset` of MNIST:\`


```r
local_folder <- './datasets/mnist_digits'
train_dataset = torchvision$datasets$MNIST(root = local_folder, 
                                           train = TRUE, 
                                           transform = transforms$ToTensor(),
                                           download = TRUE)
train_dataset
```

```
#> Dataset MNIST
#>     Number of datapoints: 60000
#>     Root location: ./datasets/mnist_digits
#>     Split: Train
#>     StandardTransform
#> Transform: ToTensor()
```

## Python objects

Sometimes we are interested in knowing the internal components of a class. In that case, we use the `reticulate` function `py_list_attributes()`.

In this example, we want to show the attributes of `train_dataset`:


```r
reticulate::py_list_attributes(train_dataset)
```

```
#>  [1] "__add__"                "__class__"              "__delattr__"           
#>  [4] "__dict__"               "__dir__"                "__doc__"               
#>  [7] "__eq__"                 "__format__"             "__ge__"                
#> [10] "__getattribute__"       "__getitem__"            "__gt__"                
#> [13] "__hash__"               "__init__"               "__init_subclass__"     
#> [16] "__le__"                 "__len__"                "__lt__"                
#> [19] "__module__"             "__ne__"                 "__new__"               
#> [22] "__reduce__"             "__reduce_ex__"          "__repr__"              
#> [25] "__setattr__"            "__sizeof__"             "__str__"               
#> [28] "__subclasshook__"       "__weakref__"            "_check_exists"         
#> [31] "_format_transform_repr" "_repr_indent"           "class_to_idx"          
#> [34] "classes"                "data"                   "download"              
#> [37] "extra_repr"             "processed_folder"       "raw_folder"            
#> [40] "resources"              "root"                   "target_transform"      
#> [43] "targets"                "test_data"              "test_file"             
#> [46] "test_labels"            "train"                  "train_data"            
#> [49] "train_labels"           "training_file"          "transform"             
#> [52] "transforms"
```

Knowing the internal methods of a class could be useful when we want to refer to a specific property of such class. For example, from the list above, we know that the object `train_dataset` has an attribute `__len__`. We can call it like this:


```r
train_dataset$`__len__`()
```

```
#> [1] 60000
```

## Iterating through datasets

### Enumeration

Given the following training dataset `x_train`, we want to find the number of elements of the tensor. We start by entering a `numpy` array, which then will convert to a tensor with the PyTorch function `from_numpy()`:


```r
x_train_r <- array(c(3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 
                  9.779, 6.182, 7.59, 2.167, 7.042,
                  10.791, 5.313, 7.997, 3.1), dim = c(15,1))

x_train_np <- r_to_py(x_train_r)
x_train_   <- torch$from_numpy(x_train_np)          # convert to tensor
x_train    <- x_train_$type(torch$FloatTensor)      # make it a a FloatTensor
print(x_train$dtype)
print(x_train)
```

```
#> torch.float32
#> tensor([[ 3.3000],
#>         [ 4.4000],
#>         [ 5.5000],
#>         [ 6.7100],
#>         [ 6.9300],
#>         [ 4.1680],
#>         [ 9.7790],
#>         [ 6.1820],
#>         [ 7.5900],
#>         [ 2.1670],
#>         [ 7.0420],
#>         [10.7910],
#>         [ 5.3130],
#>         [ 7.9970],
#>         [ 3.1000]])
```

`length` is similar to `nelement` for number of elements:


```r
length(x_train)
x_train$nelement()    # number of elements in the tensor
```

```
#> [1] 15
#> [1] 15
```

### `enumerate` and `iterate`


```r
py = import_builtins()

enum_x_train = py$enumerate(x_train)
enum_x_train

py$len(x_train)
```

```
#> <enumerate>
#> [1] 15
```

If we directly use `iterate` over the `enum_x_train` object, we get an R list with the index and the value of the `1D` tensor:


```r
xit = iterate(enum_x_train, simplify = TRUE)
xit
```

```
#> [[1]]
#> [[1]][[1]]
#> [1] 0
#> 
#> [[1]][[2]]
#> tensor([3.3000])
#> 
#> 
#> [[2]]
#> [[2]][[1]]
#> [1] 1
#> 
#> [[2]][[2]]
#> tensor([4.4000])
#> 
#> 
#> [[3]]
#> [[3]][[1]]
#> [1] 2
#> 
#> [[3]][[2]]
#> tensor([5.5000])
#> 
#> 
#> [[4]]
#> [[4]][[1]]
#> [1] 3
#> 
#> [[4]][[2]]
#> tensor([6.7100])
#> 
#> 
#> [[5]]
#> [[5]][[1]]
#> [1] 4
#> 
#> [[5]][[2]]
#> tensor([6.9300])
#> 
#> 
#> [[6]]
#> [[6]][[1]]
#> [1] 5
#> 
#> [[6]][[2]]
#> tensor([4.1680])
#> 
#> 
#> [[7]]
#> [[7]][[1]]
#> [1] 6
#> 
#> [[7]][[2]]
#> tensor([9.7790])
#> 
#> 
#> [[8]]
#> [[8]][[1]]
#> [1] 7
#> 
#> [[8]][[2]]
#> tensor([6.1820])
#> 
#> 
#> [[9]]
#> [[9]][[1]]
#> [1] 8
#> 
#> [[9]][[2]]
#> tensor([7.5900])
#> 
#> 
#> [[10]]
#> [[10]][[1]]
#> [1] 9
#> 
#> [[10]][[2]]
#> tensor([2.1670])
#> 
#> 
#> [[11]]
#> [[11]][[1]]
#> [1] 10
#> 
#> [[11]][[2]]
#> tensor([7.0420])
#> 
#> 
#> [[12]]
#> [[12]][[1]]
#> [1] 11
#> 
#> [[12]][[2]]
#> tensor([10.7910])
#> 
#> 
#> [[13]]
#> [[13]][[1]]
#> [1] 12
#> 
#> [[13]][[2]]
#> tensor([5.3130])
#> 
#> 
#> [[14]]
#> [[14]][[1]]
#> [1] 13
#> 
#> [[14]][[2]]
#> tensor([7.9970])
#> 
#> 
#> [[15]]
#> [[15]][[1]]
#> [1] 14
#> 
#> [[15]][[2]]
#> tensor([3.1000])
```

### `for-loop` for iteration

Another way of iterating through a dataset that you will see a lot in the PyTorch tutorials is a `loop` through the length of the dataset. In this case, `x_train`. We are using `cat()` for the index (an integer), and `print()` for the tensor, since `cat` doesn't know how to deal with tensors:


```r
# reset the iterator
enum_x_train = py$enumerate(x_train)

for (i in 1:py$len(x_train)) {
    obj <- iter_next(enum_x_train)    # next item
    cat(obj[[1]], "\t")     # 1st part or index
    print(obj[[2]])         # 2nd part or tensor
}
```

```
#> 0 	tensor([3.3000])
#> 1 	tensor([4.4000])
#> 2 	tensor([5.5000])
#> 3 	tensor([6.7100])
#> 4 	tensor([6.9300])
#> 5 	tensor([4.1680])
#> 6 	tensor([9.7790])
#> 7 	tensor([6.1820])
#> 8 	tensor([7.5900])
#> 9 	tensor([2.1670])
#> 10 	tensor([7.0420])
#> 11 	tensor([10.7910])
#> 12 	tensor([5.3130])
#> 13 	tensor([7.9970])
#> 14 	tensor([3.1000])
```

Similarly, if we want the scalar values but not as tensor, then we will need to use `item()`.


```r
# reset the iterator
enum_x_train = py$enumerate(x_train)

for (i in 1:py$len(x_train)) {
    obj <- iter_next(enum_x_train)    # next item
    cat(obj[[1]], "\t")               # 1st part or index
    print(obj[[2]]$item())            # 2nd part or tensor
}
```

```
#> 0 	[1] 3.3
#> 1 	[1] 4.4
#> 2 	[1] 5.5
#> 3 	[1] 6.71
#> 4 	[1] 6.93
#> 5 	[1] 4.17
#> 6 	[1] 9.78
#> 7 	[1] 6.18
#> 8 	[1] 7.59
#> 9 	[1] 2.17
#> 10 	[1] 7.04
#> 11 	[1] 10.8
#> 12 	[1] 5.31
#> 13 	[1] 8
#> 14 	[1] 3.1
```

> We will find very frequently this kind of iterators when we read a dataset read by `torchvision`. There are several different ways to iterate through these objects as you will find.

## Zero gradient

The zero gradient was one of the most difficult to implement in R if we don't pay attention to the content of the objects carrying the **weights** and **biases**. This happens when the algorithm written in **PyTorch** is not immediately translatable to **rTorch**. This can be appreciated in this example.

> We are using the same seed in the PyTorch and rTorch versions, so, we could compare the results.

### Code version in Python


```python
import numpy as np
import torch

torch.manual_seed(0)  # reproducible

# Input (temp, rainfall, humidity)
```

```
#> <torch._C.Generator object at 0x7f42c604e250>
```

```python
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')
                    

# Convert inputs and targets to tensors
inputs  = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# random weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# function for the model
def model(x):
  wt = w.t()
  mm = x @ w.t()
  return x @ w.t() + b       # @ represents matrix multiplication in PyTorch

# MSE loss function
def mse(t1, t2):
  diff = t1 - t2
  return torch.sum(diff * diff) / diff.numel()

# Running all together
# Train for 100 epochs
for i in range(100):
  preds = model(inputs)
  loss = mse(preds, targets)
  loss.backward()
  with torch.no_grad():
    w -= w.grad * 0.00001
    b -= b.grad * 0.00001
    w_gz = w.grad.zero_()
    b_gz = b.grad.zero_()
    
# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print("Loss: ", loss)    

# predictions
```

```
#> Loss:  tensor(1270.1233, grad_fn=<DivBackward0>)
```

```python
print("\nPredictions:")
```

```
#> 
#> Predictions:
```

```python
preds

# Targets
```

```
#> tensor([[ 69.3122,  80.2639],
#>         [ 73.7528,  97.2381],
#>         [118.3933, 124.7628],
#>         [ 89.6111,  93.0286],
#>         [ 47.3014,  80.6467]], grad_fn=<AddBackward0>)
```

```python
print("\nTargets:")
```

```
#> 
#> Targets:
```

```python
targets
```

```
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

### Code version in R


```r
library(rTorch)

torch$manual_seed(0)

device = torch$device('cpu')
# Input (temp, rainfall, humidity)
inputs = np$array(list(list(73, 67, 43),
                   list(91, 88, 64),
                   list(87, 134, 58),
                   list(102, 43, 37),
                   list(69, 96, 70)), dtype='float32')

# Targets (apples, oranges)
targets = np$array(list(list(56, 70), 
                    list(81, 101),
                    list(119, 133),
                    list(22, 37), 
                    list(103, 119)), dtype='float32')


# Convert inputs and targets to tensors
inputs = torch$from_numpy(inputs)
targets = torch$from_numpy(targets)

# random numbers for weights and biases. Then convert to double()
torch$set_default_dtype(torch$float64)

w = torch$randn(2L, 3L, requires_grad=TRUE) #$double()
b = torch$randn(2L, requires_grad=TRUE) #$double()

model <- function(x) {
  wt <- w$t()
  return(torch$add(torch$mm(x, wt), b))
}

# MSE loss
mse = function(t1, t2) {
  diff <- torch$sub(t1, t2)
  mul <- torch$sum(torch$mul(diff, diff))
  return(torch$div(mul, diff$numel()))
}

# Running all together
# Adjust weights and reset gradients
for (i in 1:100) {
  preds = model(inputs)
  loss = mse(preds, targets)
  loss$backward()
  with(torch$no_grad(), {
    w$data <- torch$sub(w$data, torch$mul(w$grad, torch$scalar_tensor(1e-5)))
    b$data <- torch$sub(b$data, torch$mul(b$grad, torch$scalar_tensor(1e-5)))
    
    w$grad$zero_()
    b$grad$zero_()
  })
}

# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
cat("Loss: "); print(loss)

# predictions
cat("\nPredictions:\n")
preds

# Targets
cat("\nTargets:\n")
targets
```

```
#> <torch._C.Generator>
#> Loss: tensor(1270.1237, grad_fn=<DivBackward0>)
#> 
#> Predictions:
#> tensor([[ 69.3122,  80.2639],
#>         [ 73.7528,  97.2381],
#>         [118.3933, 124.7628],
#>         [ 89.6111,  93.0286],
#>         [ 47.3013,  80.6467]], grad_fn=<AddBackward0>)
#> 
#> Targets:
#> tensor([[ 56.,  70.],
#>         [ 81., 101.],
#>         [119., 133.],
#>         [ 22.,  37.],
#>         [103., 119.]])
```

Notice that while we are in Python, the tensor operation, gradient ($\nabla$) of the weights $w$ times the **Learning Rate** $\alpha$, is:

$$w = -w + \nabla w \; \alpha$$

In Python, it is a very straight forwward and clean code:


```python
w -= w.grad * 1e-5
```

In R, without generics, it shows a little bit more convoluted:


```r
w$data <- torch$sub(w$data, torch$mul(w$grad, torch$scalar_tensor(1e-5)))
```

## R generic functions

Which why we simplified these common operations using the R generic function. When we use the generic methods from **rTorch** the operation looks much neater.


```r
w$data <- w$data - w$grad * 1e-5
```

The following two expressions are equivalent, with the first being the long version natural way of doing it in **PyTorch**. The second is using the generics in R for subtraction, multiplication and scalar conversion.


```r
param$data <- torch$sub(param$data,
                        torch$mul(param$grad$float(),
                          torch$scalar_tensor(learning_rate)))
}
```


```r
param$data <- param$data - param$grad * learning_rate
```
