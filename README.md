# A Minimal Book for rTorch

*Why do we want a package of something that is already working well, such as PyTorch?*

There are several reasons, but the main one is to bring another machine learning framework to R. Probably, it is just me but I feel *PyTorch* very comfortable to work with. Feels pretty much like everything else in Python. Very **pythonic**. I have tried other frameworks in R. The closest that matches a natural language like PyTorch, is [MXnet](https://mxnet.apache.org/versions/1.7.0/get_started?). Unfortunately, *MXnet* it is the hardest to install and maintain after updates.

Yes. I could have worked directly with *PyTorch* in a native Python environment, such as *Jupyter,* or *PyCharm,* or [vscode](https://code.visualstudio.com/docs/python/jupyter-support) notebooks but it very hard to quit **RMarkdown** once you get used to it. It is the real thing in regards to [literate programming](https://en.wikipedia.org/wiki/Literate_programming) and **reproducibility**. It does not only contribute to improving the quality of the code but establishes a workflow for a better understanding of a subject by your intended readers, in what is been called the *literate programming paradigm*.

This has the additional benefit of giving the ability to write combination of *Python* and *R* code together in the same document. There will be times when it is better to create a class in *Python*; and other times where *R* will be more convenient to handle a data structure. I show some examples using `data.frame` and `data.table` along with *PyTorch* tensors.

