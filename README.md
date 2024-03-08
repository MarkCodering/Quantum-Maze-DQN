# Official Repo: Deep Q-Learning with Hybrid Quantum Neural Network on Solving Maze Problems

## Introduction

Author: [Mark Chen](https://github.com/MarkCodering) <br>
Paper URL: [Deep Reinforcement Learning Using Hybrid Quantum Neural Network](https://arxiv.org/abs/2304.10159) <br>
Published Paper URL: [Deep Q-Learning with Hybrid Quantum Neural Network on Solving Maze Problems](link.springer.com/article/10.1007/s42484-023-00137-)
[Code Template Reference](https://github.com/giorgionicoletti/deep_Q_learning_maze)

## Research Abstract
Quantum computing holds great potential for advancing the limitations of machine learning algorithms to handle higher dimensions of data and reduce overall training parameters in deep learning (DL) models. This study uses a trainable variational quantum circuit (VQC) on a gate-based quantum computing model to investigate the potential for quantum benefit in a model-free reinforcement learning problem. Through a comprehensive investigation and evaluation of the current model and capabilities of quantum computers, we designed and trained a novel hybrid quantum neural network based on the latest Qiskit and PyTorch framework. We compared its performance with a full-classical CNN with and without an incorporated VQC. Our research provides insights into the potential of deep quantum learning to solve a maze problem and, potentially, other reinforcement learning problems. We conclude that reinforcement learning problems can be practical with reasonable training epochs. Moreover, a comparative study of full-classical and hybrid quantum neural networks is discussed to understand these two approaches’ performance, advantages, and disadvantages to deep Q-learning problems, especially on larger-scale maze problems larger than 4x4.

## Quick Start

Recommanded environment: Python 3.8.5, CUDA>11, Qiskit >=0.40, Linux Ubuntu 20.04

```shell
git clone
cd Deep-Reinforcement-Learning-using-quDNN
pip install -r requirements.txt
cd src
# Run the deep Q-learning model with hybrid quantum neural network, please select deep_q_learning_quantum.ipynb
```
