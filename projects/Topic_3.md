# Topic 3 — Mixed-Precision Inference on GPUs

## Overview

Modern GPUs provide multiple execution paths for numerical computation, including **general-purpose CUDA cores** and **specialized Tensor Cores**, each with different performance characteristics. At the same time, reducing numerical precision is a key technique for improving performance and lowering memory footprint during neural network inference.

In this project, **you should study mixed-precision inference on GPUs as a performance problem**, by implementing and comparing different execution strategies.

## Goal

The goal of this project is to **analyze the performance impact of mixed-precision inference** on GPUs.

You should implement a neural-network inference workload and evaluate how different precision formats and execution units affect:

* Runtime and throughput
* Memory bandwidth usage
* Numerical accuracy relative to an FP32 baseline

Correctness is required, but **performance analysis is the primary objective**.

## Project structure

You should choose **one of the following two options** and clearly state your choice in your proposal.



## Option 1 — Mixed Precision on CUDA Cores vs Tensor Cores

### Description

In this option, you should implement the same inference workload using **two different GPU execution paths**:

1. **CUDA cores** using custom CUDA kernels
2. **Tensor Cores** using GPU libraries (e.g., cuBLAS or cuBLASLt)

Both implementations should use mixed-precision arithmetic and be directly comparable.

### Expectations

You should:

* Implement an FP32 baseline
* Implement mixed-precision inference (e.g., FP16 or BF16 inputs with FP32 accumulation)
* Use CUDA cores for one implementation and Tensor Cores for the other
* Measure and compare performance between the two approaches

The focus should be on understanding how **specialized hardware units** change performance behavior.



## Option 2 — Mixed Precision with Row-Wise Scaling

### Description

In this option, you should focus on reducing **memory footprint and numerical range issues** by combining mixed precision with **row-wise scaling** of weights.

The idea is to factor out a scale value from each row of a weight matrix so that the remaining values can be stored in lower precision.

### Expectations

You should:

* Implement an FP32 baseline
* Implement mixed-precision inference on CUDA cores
* Apply row-wise scaling to weight matrices to reduce numerical range
* Store scaled weights in lower precision (e.g., FP16)
* Store scale factors separately and apply them during inference

You should analyze how scaling affects:

* Memory footprint
* Performance
* Numerical accuracy



## Workload scope

You should use a **simple inference workload** such as:

* A single-layer or two-layer MLP
* A dense linear layer followed by a nonlinearity

Training is **not required**. The workload should be chosen to make performance differences clear and measurable.

## What “from scratch” means

You should implement:

* CUDA kernels for the core inference computation
* Precision handling and data type conversions
* A dense FP32 baseline for comparison

You may use:

* CUDA runtime and standard CUDA libraries
* GPU libraries for Tensor Core execution (Option 1)
* Simple CPU-side preprocessing and validation

You should not rely on high-level machine learning frameworks for computation.

## Proposal requirement

Before starting implementation, **you should submit a short proposal (1–2 pages)** describing:

* Which option you chose
* The inference workload you will implement
* Precision formats to be evaluated
* Your CUDA execution strategy
* Performance metrics you will collect

You should **present this proposal**, and if it is **approved**, you may proceed with the implementation.

## Evaluation and reporting

Your final report should include:

* Description of the implementation
* Performance comparison against the FP32 baseline
* Analysis of precision-related tradeoffs
* Discussion of GPU execution behavior

Your work should demonstrate a clear understanding of how mixed precision and GPU hardware features affect inference performance.
