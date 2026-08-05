# Continual Learning CNN Report


## Introduction

Neural networks usually forget previous knowledge when trained on new tasks.
This problem is called catastrophic forgetting.



## Dataset

The experiments use:

- MNIST
- Permuted MNIST



## Baseline Experiment

The model learns Task 1 and then Task 2.

After learning Task 2, performance on Task 1 decreases.



## Rehearsal Method

A small memory buffer stores examples from previous tasks.

During training on new tasks, old examples are replayed.



## EWC Method

Elastic Weight Consolidation prevents important parameters from changing too much.



## Evaluation

Metrics:

- Task accuracy
- Average accuracy
- Forgetting measure



## Conclusion

Continual learning methods improve the ability of neural networks to retain previous knowledge.