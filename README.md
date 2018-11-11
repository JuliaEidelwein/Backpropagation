# Backpropagation
A neural network made from scratch.
### Running commands
Currently, the command line to call execute this neural network is:
```sh
$ python ./main.py network.txt initial_weights.txt dataset.txt
```
Where _network.txt_ is a file which content determines the topology (number or neurons and layers) of the network, the first line denotes the value of the regularization parameter (_lambda_). The example above defines a neural network with λ = 0.25 and 3 layers, containing 2, 3 and 1 neurons respectively, excluding the bias neuron.
>0.25
2
3
1

And _initial_weights.txt_, as said by its name, gives the initial weights to the layers.
