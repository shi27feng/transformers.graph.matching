# transformers.graph.matching

Applying the cross-attention/similarity matrix of Transformer Architecture to graph matching problem

`matching` is not necessary to be exact, but much close sometime is good enough.

### Baseline codes are reproduced and based on:
- [SimGNN (PyTorch, extended version)](https://github.com/gospodima/Extended-SimGNN), [SimGNN (TensorFlow)](https://github.com/yunshengb/SimGNN)
- [GMNN](https://github.com/deepmind/deepmind-research/tree/master/graph_matching_networks)

### Graph Wavelet Neural Network can be an alternative to _GCN_
- [Graph Wavelet Neural Networks](https://github.com/benedekrozemberczki/GraphWaveletNeuralNetwork)

### To extend to very large graphs, we consider to coarsen the graph, but there's no perfect algorithm, we can try a few approximation algorithms:
- [Graph Coarsening (code example)](https://github.com/loukasa/graph-coarsening) and [`blog`](https://andreasloukas.blog/2018/11/05/multilevel-graph-coarsening-with-spectral-and-cut-guarantees/)


