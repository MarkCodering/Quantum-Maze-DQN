# Deep Reinforcement Learning Using Hybrid Quantum Neural Network

## Introduction

Author: [Mark Chen](https://github.com/MarkCodering) <br>
Paper URL: [Deep Reinforcement Learning Using Hybrid Quantum Neural Network](https://arxiv.org/abs/2304.10159) <br>
[Code Template Reference](https://github.com/giorgionicoletti/deep_Q_learning_maze)

## Research Abstract

Quantum computing holds great potential for advancing the limitations of machine learning algorithms to handle higher data dimensions and reduce overall training parameters in deep neural network (DNN) models. This study uses a parameterized quantum circuit (PQC) on a gate-based quantum computer to investigate the potential for quantum advantage in a model-free reinforcement learning problem. Through a comprehensive investigation and evaluation of the current model and capabilities of quantum computers, we designed and trained a novel hybrid quantum neural network based on the latest Qiskit and PyTorch framework. We compared its performance with a full-classical DNN with and without an integrated PQC. Our research provides insights into the potential of deep quantum learning to solve a maze problem and, potentially,  other reinforcement learning problems. We conclude that various reinforcement learning problems can be effective with reasonable training epochs. Moreover, a comparative discussion of the various quantum reinforcement learning model on maze problems is discussed to evaluate our research's overall potential and advantages.

## Quick Start
Recommanded environment: Python 3.8.5, CUDA>11, Qiskit >=0.40, Linux Ubuntu 20.04
```shell
git clone
cd Deep-Reinforcement-Learning-using-quDNN
pip install -r requirements.txt
cd src 
```