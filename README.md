# A11_starter

note: we skipped A10 (= A9/B)

Assignment 11: ANN XML/API
---------------------------

Understand the differences between API, Library, and Framework. (Are there any?)
Take a look at the Facade design pattern [GOF].

Design an Application Programming Interface (API) to create and configure an artificial neural network (ANN).
Write a Java Library implementing this API. 

You can make it simple, i.e., stick to the assumptions below. However, spend a few minutes to think about what it would mean to make the system more flexible (for example, the layers are potentially not fully connected).

Assumptions and requirements: 
1. The ANN is layered (configurable number of layers)
1. Each layer has a configurable number of neurons
1. Layers are fully connected
1. There is one activation function per layer
1. Optional: A neuron may have a separate activation function (this 'overrules' the activation function of the layer)
1. The initial values should be (somehow?!) configurable. First, think about why. Then think about what makes sense and how to realize this in Java (you can support multiple ways of configuration).

Design an XML-format to represent an ANN configuration and write a sample XML file.

Write an XML parser that reads in your configuration and that uses your API to create the network.

Not part of this assignment, but worth thinking about:
Look at your implementation of assignment A9B to get hints about what should be part of the ANN description.
Could one rewrite the A9B assignment such that there is no problem specific code, but everything you need is part of a (bigger) XML specification format?
