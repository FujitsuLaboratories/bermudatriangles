# Datasets (Tasks)
This code is a supplemenary of [BERMUDA TRIANGLES: GNNS FAIL TO DETECT
SIMPLE TOPOLOGICAL STRUCTURES](https://openreview.net/pdf?id=Vz_Nl9MSQnu)
## Triangle Presense

In this task we generate relatively complex graphs which contain either exactly 1 or exactly 0
cycles of length 3 (triangle structures).

## Clique Distance

In this task we generate a BA graph with two cliques attached.
We assign class 1 to graphs where the distance between cliques is larger than a threshold and class 0 othrewise.

# Structure

* `scripts` folder contains code for creation of datasets and converting to other formats.
See the readme inside the folder for more information.
* `triangles` folder contains triangles dataset: 10k training data, 1k test data and 1k validation data.
* `clique` folder contains clique distance dataset with the same data distribution as clique distance.
* `images` folder contains visualizations of graphs (non-filtered)

# Data Format

Both datasets are stored in our internal format.
It consists of two files: sp (adjacency matrix) and cl (graph classification labels).
We provide a pair of these files for each split.
Script folder also contains a converter to [TU dataset format](https://chrsmrrs.github.io/datasets/).

## SP file

SP file contains adjacency matrices for each data set.
It is a tab-separated file where each line correspond to an edge in a graph.
The SP file has the following columns:

* Graph ID (arbitrary string)
* Start Node ID
* End Node ID
* Edge weight (always 1)

Graphs corresponding to each ID must be stored successively without interleaving with other graphs,
but the order of edges can be arbitrary.

## CL file

It is also a tab-separated file with two columns:

* Graph ID
* Graph Label

The order of entries must be the same with the SP file.

# Citation
```
@inproceedings{
tolmachev2021bermuda,
title={Bermuda Triangles: {GNN}s Fail to Detect Simple Topological Structures},
author={Arseny Tolmachev and Akira Sakai and Masaru Todoriki and Koji Maruhashi},
booktitle={ICLR 2021 Workshop on Geometrical and Topological Representation Learning},
year={2021},
url={https://openreview.net/forum?id=Vz_Nl9MSQnu}
}
```

# License Information

All data is licensed under CC0.
All code is licensed under MIT license. (Or change it to whatever you want)
