# NNI Student Program 2020: Task  2.2 Tutorial

Welcome to NNI Student Program 2020!

NNI Student Program 2020 is an online program focused on introducing students to use NNI to manage their automated machine learning (AutoML) experiments. Students could collectively advance automating machine learning experiment and development alongside the NNI community while providing open source code written and released to benefit their scientific research programs. 

For more details of the program, please visit our **registration page** (TBD).

## Introduction

In this program, you are going to train a deep learning model that predicts the labels for CIFAR-10 dataset. If you are not familiar with this task, or deep learning, it's recommended that you should follow step 0.1 first. You can also start directly from step 1, which demands you to tune a deep learning model with pre-defined architecture. In step 2, we will demonstrate the possibility to tune the architecture, aka, NAS, which will hopefully further boost the performance.

We encourage students at all levels to become developer and contributor of NNI, as long as you are pro-active learning and devoting.

Ready? Let's start our machine learning journey with NNI.

## Step 0.1

First we'll get warmed-up. I will skip this part because everything is written well and clear in the docs of PyTorch.

For more details of step 0.1, please visit [PyTorch tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Step 1

In this step, you will tune the `main.py` in `1-hpo`. You goal is simple: you should make the final accuracy reported from the code as high as possible.

You can get started already. In case still confused, I'll give some hints here. If you directly run `main.py`, you might not get an impressive performance. But if you run it with some smart arguments (parameters), you will actually do much better. When you type `python main.py -h`, you will see a bunch of options for running this code. The question is, how can you know which parameter set is best suited for this task?

This is where NNI can be helpful. If you are not a parameter-tuning expert who has the instinct to pick some parameters that definitely works as a charm, leveraging the power of NNI and making it choose for you is a smart option.

For more details of **running HPO with NNI**, please visit HPO [Document](https://nni.readthedocs.io/zh/latest/hyperparameter_tune.html)

After you are done with this step, you should also learn some details of all the models used in this step, which will get you ready for the next step.

## Step 2

In step 2, we are trying to design a model on our own. To be specific, it's to design a model automatically. We'll start with search space of DARTS, which is shown in `2-nas/model.py`.

The first thing you should try is to integrate the training code from step 1, select a random architecture in the search space, and verify the performance.

Then refer to NNI docs for all the NAS trainers you can use for this task...

For more details of **running NAS with NNI**, please visit NAS [Document](https://nni.readthedocs.io/zh/latest/nas.html)

## Tips

For more details of **NNI Tutorial**, please visit our Github Community: https://github.com/microsoft/nni

You are encouraged to regular check-ins with your mentor, listen and respond to their feedback would be helpful.

Looking forward to your good news!