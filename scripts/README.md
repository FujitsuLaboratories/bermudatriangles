# Contents

This folder contains scripts for generating the datasets and a converter to a TU format.

# Dependencies

* Python 3.7+
* scikit-learn
* numpy
* networkx

Non-packages (optional)

* graphviz
* ghostcript

# Generation Scripts

## Triangles

We generate points on 2D plane, and then connect either
k nearest ones in a graph or every point in a specific radius.

After that we remove edges from graphs which belong to at least one triangle,
prioritizing edges which belong to multiple triangles until only 1 or 0 triangles remain.

The main executable file is `gen_triangle.py`.
Basic Execution is:

```
python3 gen_triangle.py --out=tri-test --num-data=10000
```

It is also possible to use random initial graph instead of proximity-based one:

```
python3 gen_triangle.py --out=tri-test --num-data=10000 --dist-matrix=random
```

It is possible to produce the visualization of all graphs (uses graphviz) by using
`--graphviz=/path/to/out` parameter.
It will produce a pdf file for each graph.
`--graphviz-merge` will also merge all pdfs into a single multi-page pdf (uses ghostscript).


## Clique Distance

For this dataset we generate a base BA-graph and attach two clique structures to random nodes of that graph.
We decide the class of the graph based on a distance between two graphs.

The main executable is `gen_clique.py`.
Basic execution is:

```
mkdir cl-test
python3 gen_clique.py -n 1000 -bn 10,20 -cn 4 -th 4,4 -sp cl-test/all_sp.txt -cl cl-test/all_cl.txt
```

Passing the `--debug` parameter will produce a pdf file with the graph visualizations.

# Undermanned Logistic Regression filtering

We filter datasets by removing data items which are easy to guess.
We use logistic regression classifier with a feature set which is not enough to solve the task at hand
and cross-validation with sliding window.

Basic execution is: (after generating triangle data as shown above)
```
python3 lr_filter.py --root tri-test --n_folds=10 --train_folds=7
```
It will output something like:
```
Loaded 10000 training examples...
fold 0 apr=65.10/78.72/74.41
fold 1 apr=66.37/80.04/73.06
fold 2 apr=67.67/83.95/75.26
fold 3 apr=65.33/82.29/73.85
fold 4 apr=65.37/84.31/75.34
fold 5 apr=63.43/80.11/75.38
fold 6 apr=63.37/75.75/73.29
fold 7 apr=62.80/77.20/73.76
fold 8 apr=63.83/79.70/73.10
fold 9 apr=64.50/78.67/72.20
hits stats= [6593, 520, 396, 2491]
```

10 lines are accuracy, precision (at threshold), recall (at threshold) of each cross-validation fold.
The last line is how much examples were answered correctly N times (with high confidence).
In this case 2491 examples were answered correctly 3 times, which means that those examples are easily recognisable without solving the task at hand at all.

Input data is specified using `--root` and `--prefix` (default=`train_`) parameters,
to use `<prefix>{sp,cl}.txt` files.

We use `--n_folds=10 --train_folds=6` for the filtering of our datasets.

By default the script only computes LR performance and cross-validation hit stats.
The parameter `--out=/output/path` enables filtering and specifies the path to write filtered examples.
It is possible to control the number of filtered examples with `--target_cnt` parameter.

It is also possible to view trained feature weights with the `--dump_features=path` option.

Generally, we generated ~200k raw examples and produced 10k train, 1k test and 1k validation
data by using `--traget_cnt=12000`.

# Convertion

`dt2tu.py` script converts our format to TU Dataset format, optionaly zipping the output files.
See the script for details on how to use it.

Example: converting clique training data to the TU format
```
python3 dt2tu.py ../clique/ cliq-tu --dt-prefix=train_
```

# Remarks

Provided datasets were produced by slightly different versions of generation/filtering scripts and
it will not be possible to generate exactly same versions of the datasets,
but newly generated datasets will have the same properties.