# Overview

## `01_backprop_weight_evolution.py`
Shows that weights change systematically when using backpropagation:
for some steps the changes are into the same direction.
Plots will be generated and can be found in the folder `weight_evolution_plots`.

Example usage:
`python 01_backprop_weight_evolution.py`



## `02_backprop_vs_deltaw_for_a_mlp.py`
Shows that a simple weight change rule (I call it *DeltaW* here) works.
It can be used to optimize the weights of an MLP, but it’s much slower than backpropagation.

DeltaW is a gradient-free learning algorithm that updates neural network weights by adding random perturbations (delta vectors) to each parameter.
After each update, if the loss decreases, the changes are kept and the same delta vectors are retained for the next iteration; if the loss increases, the changes are reverted and new random deltas are sampled.

Example usage:
`python 02_backprop_vs_deltaw_for_a_mlp.py --train 5000 --test 1000 --epochs 1000 --batch 256 --hidden 64`


## `03_backprop_vs_deltaw_for_a_cnn.py`
Shows that *DeltaW* completely fails with optimizing the weights for a CNN.
The number of parameters is simply too large in order to find a good random weight change vector.

Example usage:
`python 03_backprop_vs_deltaw_for_a_cnn.py --dataset synthetic --n_samples 1000 --classes 5 --epochs 50 --device cpu`
