# (PART) Getting Started {.unnumbered}

# Introduction {#intro}

*Last update: Sun Oct 25 13:00:41 2020 -0500 (265c0b3c1)*



## Motivation

*Why do we want a package of something that is already working well, such as PyTorch?*

There are several reasons, but the main one is to bring another machine learning framework to R. Probably, it is just me but I feel *PyTorch* very comfortable to work with. Feels pretty much like everything else in Python. Very **pythonic**. I have tried other frameworks in R. The closest that matches a natural language like PyTorch, is [MXnet](https://mxnet.apache.org/versions/1.7.0/get_started?). Unfortunately, *MXnet* it is the hardest to install and maintain after updates.

Yes. I could have worked directly with *PyTorch* in a native Python environment, such as *Jupyter,* or *PyCharm,* or [vscode](https://code.visualstudio.com/docs/python/jupyter-support) notebooks but it very hard to quit **RMarkdown** once you get used to it. It is the real thing in regards to [literate programming](https://en.wikipedia.org/wiki/Literate_programming) and **reproducibility**. It does not only contribute to improving the quality of the code but establishes a workflow for a better understanding of a subject by your intended readers [@knuth1983], in what is been called the *literate programming paradigm* [@cordes1991].

This has the additional benefit of giving the ability to write combination of *Python* and *R* code together in the same document. There will be times when it is better to create a class in *Python*; and other times where *R* will be more convenient to handle a data structure. I show some examples using `data.frame` and `data.table` along with *PyTorch* tensors.

## Start using `rTorch`

Start using `rTorch` is very simple. After installing the minimum system requirements -such as *conda* -, you just call it with:


```{.r .badCode}
library(rTorch)
```

There are several ways of testing if `rTorch` is up and running. Let's see some of them:

### Get the PyTorch version


```r
rTorch::torch_version()
```

```
#> [1] "1.6"
```

### PyTorch configuration

This will show the PyTorch version and the current version of Python installed, as well as the paths to folders where they reside.


```r
rTorch::torch_config()
```

```
#> PyTorch v1.6.0 (~/miniconda3/envs/r-torch/lib/python3.7/site-packages/torch)
#> Python v3.7 (~/miniconda3/envs/r-torch/bin/python)
#> NumPy v1.19.4)
```

------------------------------------------------------------------------

## What can you do with `rTorch`

Practically, you can do everything you could with **PyTorch** within the **R** ecosystem. Additionally to the `rTorch` module, from where you can extract methods, functions and classes, there are available two more modules: `torchvision` and `np`, which is short for `numpy`. We could use the modules with:


```r
rTorch::torchvision
rTorch::np
rTorch::torch
```

```
#> Module(torchvision)
#> Module(numpy)
#> Module(torch)
```

## Getting help

We get a glimpse of the first lines of the `help("torch")` via a Python chunk:


```python
help("torch")
```

```
...
#> NAME
#>     torch
#> 
#> DESCRIPTION
#>     The torch package contains data structures for multi-dimensional
#>     tensors and mathematical operations over these are defined.
#>     Additionally, it provides many utilities for efficient serializing of
#>     Tensors and arbitrary types, and other useful utilities.
...
```


```python
help("torch.tensor")
```

```
...
#> Help on built-in function tensor in torch:
#> 
#> torch.tensor = tensor(...)
#>     tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False) -> Tensor
#>     
#>     Constructs a tensor with :attr:`data`.
#>     
#>     .. warning::
#>     
#>         :func:`torch.tensor` always copies :attr:`data`. If you have a Tensor
#>         ``data`` and want to avoid a copy, use :func:`torch.Tensor.requires_grad_`
#>         or :func:`torch.Tensor.detach`.
#>         If you have a NumPy ``ndarray`` and want to avoid a copy, use
#>         :func:`torch.as_tensor`.
#>     
#>     .. warning::
#>     
#>         When data is a tensor `x`, :func:`torch.tensor` reads out 'the data' from whatever it is passed,
#>         and constructs a leaf variable. Therefore ``torch.tensor(x)`` is equivalent to ``x.clone().detach()``
#>         and ``torch.tensor(x, requires_grad=True)`` is equivalent to ``x.clone().detach().requires_grad_(True)``.
...
```


```python
help("torch.cat")
```

```
...
#> Help on built-in function cat in torch:
#> 
#> torch.cat = cat(...)
#>     cat(tensors, dim=0, out=None) -> Tensor
#>     
#>     Concatenates the given sequence of :attr:`seq` tensors in the given dimension.
#>     All tensors must either have the same shape (except in the concatenating
#>     dimension) or be empty.
#>     
#>     :func:`torch.cat` can be seen as an inverse operation for :func:`torch.split`
#>     and :func:`torch.chunk`.
#>     
#>     :func:`torch.cat` can be best understood via examples.
#>     
#>     Args:
#>         tensors (sequence of Tensors): any python sequence of tensors of the same type.
#>             Non-empty tensors provided must have the same shape, except in the
#>             cat dimension.
#>         dim (int, optional): the dimension over which the tensors are concatenated
#>         out (Tensor, optional): the output tensor.
...
```


```python
help("numpy.arange")
```

```
...
#> Help on built-in function arange in numpy:
#> 
#> numpy.arange = arange(...)
#>     arange([start,] stop[, step,], dtype=None)
#>     
#>     Return evenly spaced values within a given interval.
#>     
#>     Values are generated within the half-open interval ``[start, stop)``
#>     (in other words, the interval including `start` but excluding `stop`).
#>     For integer arguments the function is equivalent to the Python built-in
#>     `range` function, but returns an ndarray rather than a list.
#>     
#>     When using a non-integer step, such as 0.1, the results will often not
#>     be consistent.  It is better to use `numpy.linspace` for these cases.
#>     
#>     Parameters
#>     ----------
#>     start : number, optional
#>         Start of interval.  The interval includes this value.  The default
#>         start value is 0.
#>     stop : number
#>         End of interval.  The interval does not include this value, except
#>         in some cases where `step` is not an integer and floating point
#>         round-off affects the length of `out`.
#>     step : number, optional
...
```

Finally, these are the classes for the module `torchvision.datasets`. We are using Python to list them using the `help` function.


```python
help("torchvision.datasets")
```

```
...
#> Help on package torchvision.datasets in torchvision:
#> 
#> NAME
#>     torchvision.datasets
#> 
#> PACKAGE CONTENTS
#>     caltech
#>     celeba
#>     cifar
#>     cityscapes
#>     coco
#>     fakedata
#>     flickr
#>     folder
#>     hmdb51
#>     imagenet
#>     kinetics
#>     lsun
#>     mnist
#>     omniglot
#>     phototour
#>     samplers (package)
#>     sbd
#>     sbu
#>     semeion
#>     stl10
#>     svhn
#>     ucf101
#>     usps
#>     utils
#>     video_utils
#>     vision
#>     voc
#> 
#> CLASSES
...
```

In other words, all the functions, modules, classes in PyTorch are available to rTorch.
