ArtificialNeuralNetwork
=======================

An implementation of a (initially) fully-connected, feed-forward artificial neural network written in Java utilizing iRPROP+ training algorithm.

##Implementation Details

###iRPROP+
iRPROP+, or improved Resilient Propogation with Weight Backtracking, is a first-order optimization algorithm used for supervised learning of artificial neural networks.  Based on Rprop developed by Martin Riedmiller and Heinrich Braun in 1992, iRPROP+ was created by Christian Igel and Michael Hüsken in 2000 as an improved variant.

The Rprop family of error optimization takes into account the sign of the partial derivative over each pattern, updating all weights independently.

####References
Rprop+: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417

iRPROP+: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.1332
