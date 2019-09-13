# Graph Element Networks
Code for the [paper](http://proceedings.mlr.press/v97/alet19a.html) Graph Element Networks: adaptive, structured computation and memory. See [here](https://arxiv.org/abs/1904.09019) for the arxiv link.

You can see the talk summarizing the paper [here](https://www.youtube.com/watch?v=wp9CjkOQm48).
## Simplified and faster code for Poisson experiments
Since the paper, we have simplified the code, now relying on [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric), which also optimizes the runtime of GNNs.

Now the code is faster and **easy to customize**. In particular, the [GEN](https://github.com/FerranAlet/graph_element_networks/blob/master/GEN.py) file contains the class that can be customized, as shown in [GENSoftNN](https://github.com/FerranAlet/graph_element_networks/blob/master/gen_softnn.py).

Despite a couple minor changes to make the code faster and runnable inside a Docker container (see the Dockerfile), we checked that results are very similar to the original ones.

You can download the image files to visualize the node optimization experiments [here](https://drive.google.com/drive/folders/1T02Imopghpxg8ajiVw5jjrcj2pY2EpS_?usp=sharing).
## Code for the scene representation experiments
You can find the self-contained code for the scene representation experiments in the [scene_representation folder](https://github.com/FerranAlet/graph_element_networks/tree/master/scene_representation) folder and the Graph Element Network code in the [composer](https://github.com/FerranAlet/graph_element_networks/blob/master/scene_representation/GEN/composer.py) and the representation function [here](https://github.com/FerranAlet/graph_element_networks/blob/master/scene_representation/GEN/structure.py#L63). Note that this part of the codebase contains the original for GENs.
