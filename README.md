# Backpropagation

A neural network made from scratch with Python3 ❤️ 🐍.❤️

* * *

## Running commands

Currently, the command line to call execute this neural network is:

```sh
/$ python3 main.py network.txt initial_weights.txt dataset.txt
```

Where `network.txt` is a file which content determines the topology (number
or neurons and layers) of the network, the first line denotes the value
of the regularization parameter (`λ`). The example above defines a
neural network with λ = 0.25 and 3 layers, containing 2, 3 and 1 neurons
respectively, excluding the bias neuron.

```txt
0.25
2
3
1
```

And `initial_weights.txt`, as said by its name, gives the initial weights to
the layers.
