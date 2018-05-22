# tensorflow-multi-gpu
A useful code sample to implement multi-gpu with tensorflow

## Introduction
This repo will illustrate the basic idea of multi-gpu implementation with tensorflow and give a general sample for users.

## Illustration
Basic idea is to replicate model for several copys and build them on GPU. Then since we need to update the parameters by applying gradients, we gather those gradients and apply their mean to all replicas.
![](https://github.com/GitBoSun/tensorflow-multi-gpu/blob/master/tf_multi_gpu.png)
Here are somethings we should pay attention to:
1. All the variables and operation to apply gradients is based on CPU while building models and calculating gradients are based on GPU.
2. You need to use **"tf.get_variable()"** to build variables rather than "tf.Variable()" becasue there is a concept of scope in "tf.get_variable()". Since you share variables across replicas, you need to use the critique function: **tf.variable_scope().reuse_variables()**. So all your model copys share parameters and will be updated at the same time. 
3. You can either split your training batch to feed one batch to several models or feed every model a batch data. 
4. You need to state the optimizer inside variable scope and calculate gradients outside scope because it can't find shared variabled related to optimizer inside scope.

## Implementation
Here I use a trick of **collection**. When building models on GPU, we add each model and gradient to collections. Then when we want to use them to fees data and apply gradients, just pull them up. I think this is the easiest way to implement multi-gpu in tensorflow.
