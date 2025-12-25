# Topic 1 — Training a CNN From Scratch Using CUDA

## Goal

In this project, **you should implement and train a Convolutional Neural Network (CNN) entirely using CUDA**, with a **primary emphasis on performance**. While functional correctness is required (the network must learn and improve accuracy), the main objective is to analyze, reason about, and optimize how the workload maps onto GPU hardware.

You should treat this project as a **GPU systems problem first**, and a machine learning problem second.

## Performance-Centered Focus

You should focus on both:

* **Functional correctness** — the model trains correctly and the loss decreases over time.
* **Performance reasoning (primary focus)** — how efficiently your CUDA implementation uses GPU resources.

Your design decisions should be justified in terms of:

* Kernel launch configuration (threads, blocks, grid shape)
* Memory access patterns and bandwidth utilization
* Arithmetic intensity and data reuse
* Kernel execution time and bottlenecks

## What “from scratch” means

You should write CUDA code for the **core training pipeline**, including:

* Forward pass operations (convolution, activation, pooling, fully connected layers, softmax)
* Loss computation (e.g., cross-entropy)
* Backpropagation for gradient computation
* Weight update (e.g., SGD)

You may use:

* **Simple preprocessing on the CPU** (normalization, reshaping, batching)
* CPU-side dataset loading
* CUDA runtime and standard CUDA libraries
* cuRAND (optional) for weight initialization

You should **not** use deep learning frameworks (e.g., PyTorch, TensorFlow) for any model computation.

## Model scope

You should implement a **small but complete CNN** that can be trained on a single GPU. Suitable datasets include:

* **MNIST** (recommended for validating correctness quickly)
* **CIFAR-10** (recommended for stronger performance analysis)

The network architecture is your choice, but it should be complex enough to expose **meaningful GPU performance behavior**, especially in convolutional layers.

## Background resources (highly recommended)

To understand what training a neural network “from scratch” actually involves at an algorithmic level, you are encouraged to watch:

* Intuition for what neural networks are really computing and how learning works:
  [https://www.youtube.com/watch?v=QzY57FaENXg](https://www.youtube.com/watch?v=QzY57FaENXg)

* How a neural network is built and trained step by step from scratch:
  [https://www.youtube.com/watch?v=w8yWXqWQYmU](https://www.youtube.com/watch?v=w8yWXqWQYmU)

These resources focus on **algorithmic and mathematical intuition**, which you will then translate into efficient GPU kernels.

## Proposal requirement

Before starting implementation, **you should submit a short proposal (1–2 pages)** describing:

* The dataset you will use
* The CNN architecture you plan to implement
* Which components will be implemented in CUDA
* Where you expect performance bottlenecks to arise
* Which GPU optimizations you plan to explore
* How you will evaluate both correctness and performance

You should **present this proposal** and explain your design and performance assumptions.
If the proposal is **approved**, you may proceed with the full implementation.

## Implementation expectations

Once approved, you should:

* Implement CUDA kernels for all major training components
* Run multi-epoch training and demonstrate decreasing loss
* Measure and report kernel-level performance
* Use profiling tools to identify bottlenecks

Your implementation should clearly reflect **performance-aware design choices**, not just functional correctness.

## Evaluation and reporting

Your final report should include:

* Training curves (loss and/or accuracy)
* Detailed performance measurements (kernel timings, throughput, or occupancy-related observations)
* A discussion of where execution time is spent and why
* An explanation of at least one performance-driven design decision

Your work should demonstrate a clear understanding of how CNN training workloads interact with GPU architecture, and how CUDA programming choices affect performance.
