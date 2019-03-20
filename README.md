# Backpropagation

A neural network made from scratch with Python3 ❤️ 🐍.❤️

* * *

## Running commands

The command line to execute this neural network is:

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

`initial_weights.txt`, as said by its name, gives the initial weights to
the layers. For the topology above, a valid file would look like the following:

```txt
0.4,0.1,0.5;0.3,0.2,0.4;0.1,0.25,0.7
0.7,0.5,0.6,0.26
```

Using both files given above, the resultant network would look like as the image above. Here, the bias neurons are orange (they are used at every layer, with fixed weight 1, in order to allow a fitting function that does not cross the origin). The remaining neurons at input, hidden and output layer are pink, blue and green, respectively.

![Example Network](https://github.com/JuliaEidelwein/Backpropagation/blob/master/Example_network.png)

Lastly, the dataset must be represented with a line per instance, separating each feature with a comma (**,**). Then, a semicolon (**;**) should be added before the expected value (if there are more than one output, each of them should be separated with commas). The following example has two instances, both with two attributes and one expected value. It matches the network we specified before.

```txt
10,4;0.3
12.2,6;0.75
```
