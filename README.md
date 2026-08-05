# Continual Learning CNN

## Objective

Implement and compare methods to reduce catastrophic forgetting in CNNs.

The model learns multiple tasks sequentially without retraining from scratch.


## Dataset

### Task 1
MNIST handwritten digits


### Task 2
Permuted MNIST

Images are randomly shuffled using a fixed permutation.



## Model

A Convolutional Neural Network:

- Conv Layer
- Conv Layer
- Fully Connected Layers



## Continual Learning Methods


### 1. Baseline

Sequential training without any protection.

Shows catastrophic forgetting.



### 2. Rehearsal

Stores examples from previous tasks and replays them during new task learning.



### 3. EWC

Elastic Weight Consolidation.

Protects important parameters using Fisher Information.



## Metrics

- Accuracy
- Average Accuracy
- Forgetting Measure



## Run


Install dependencies:
