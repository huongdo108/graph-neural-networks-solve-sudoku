# Graph neural networks-Recurrent relational network

## Overview

The goal of this repository is to get familiar with graph neural networks.

In this repository, a simplified version of the recurrent relational network proposed in [this paper](http://papers.nips.cc/paper/7597-recurrent-relational-networks.pdf) is implemented.

## Dataset

The dataset is a set of sudoku puzzles with solutions obtained from [here](https://github.com/locuslab/SATNet).

Let's take a look at the dataset

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/data_visualization.PNG" align="centre">


## Graph

Each sudoku puzzle is described as a graph in which each node corresponds to one of the 9*9=81 cells in the puzzle. Each node is connected to 8+8+8-4=20 other nodes:
* 8 nodes that correspond to other cells in the same row
* 8 nodes that correspond to other cells in the same column
* 8 nodes that correspond to other cells in the same $3\times 3$ box
* 4 cells appear both in a $3\times 3$ box and in the same row or column.

In this repository, a graph is described as a set of edges. Each edge is a pair (`src_id`, `dst_id`) where `src_id` is the id of the source node and `dst_id` is the id of the destination node. Node ids are between 0 and 80.

A collate function is implemented with this graph architecture to transform the dataset into dataloader which will be feeded in the neural network.


## Graph neural network

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/recurrent_rn.png" align="centre">

The graph neural network's forward function consists of `n_iters` iterations with the following steps:
* For each node, compute the messages from all its neighbors using the message network (see description below).
* For each destination node, aggregate (by summation) all the messages sent by its neighbors. You may find it useful to use function [`index_add_`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.index_add_) for that.
* The aggregated messages are concatenated with the node inputs to form the inputs of the gated recurrent units (GRUs) whose states represent the states of the nodes. The node inputs are one-hot coded digits 1-9 of the sudoku puzzle, a missing digit is encoded with all zeros. 
* The states of the GRUs are updated using the standard GRU computations.
* The states of each graph node are linearly combined to compute the output of the corresponding node at the current iteration.

Since all graphs in the mini-batch are combined into a single graph using `collate()` function, batches can be ignored in the implementation of Graph neural network.**

**Message network**

* All messages are computed with the same message network (shared parameters).
* The message network takes as inputs the states of the source node and the states of the destination node and produces a vector with `n_edge_features`.
* A multilayer perceptron (MLP) network is used as the message network with the following architecture:
  * hidden layer with 96 neurons and ReLU nonlinearity
  * hidden layer with 96 neurons and ReLU nonlinearity
  * output layer with `n_edge_features` output features.

## Training result

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/training_result.PNG" align="centre">

## Model evaluation

Visualize graph iteration for one of the puzzles

Iteration 1

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/solved_puzzle_iter1.PNG" align="centre">

Iteration 2

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/solved_puzzle_iter2.PNG" align="centre">

Iteration 3

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/solved_puzzle_iter3.PNG" align="centre">

Iteration 4,5,6,7

<img src="https://github.com/huongdo108/graph-neural-networks-solve-sudoku/blob/master/images/solved_puzzle_iter4.PNG" align="centre">