---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

### Higher-Order Spatiotemporal Correlations

* *[Convolution goes higher-order: a biologically inspired mechanism empowers image classification](https://arxiv.org/abs/2412.06740)* - NeurIPS 2025
  * Demonstrated that incorporating multiplicative interactions between filter outputs significantly improves performance on standard computer vision benchmarks
  * Developed a novel architecture that captures complex dependencies in visual data that standard CNNs miss

* *[Higher-Order Convolution Improves Neural Predictivity in the Retina](https://arxiv.org/pdf/2505.07620)* - arXiv 2025 (presented at Bernstein 2025 & European Retina Meeting 2025)
  * Extended the higher-order convolution framework to neural response prediction in the retina
  * Showed improved predictive performance while maintaining model interpretability



### Information Decomposition and Generative Models

* *[What's Inside Your Diffusion Model? A Score-Based Riemannian Metric to Explore the Data Manifold](https://arxiv.org/abs/2505.11128)* - arXiv 2025
  * Introduced a mathematically rigorous framework for analyzing the geometry of visual representations using metrics derived from the score function in diffusion models
  * Exploited the Riemannian geometry of the learned image manifold to provide tools for understanding meaningful transformations in visual space

* *[Decomposing stimulus-specific sensory neural information via diffusion models](https://arxiv.org/abs/2505.11309)* - Spotlight (top 3%) at NeurIPS 2025
  * Established an axiomatic framework for pixel-wise and feature-wise information decomposition using score functions from diffusion models

* *A multi-scale information geometry reveals the structure of mutual information in neural populations* - Submitted to NeurIPS 2026
  * Derived a unique metric whose mean trace equals the mutual information carried by a population code, emerging from a few coarse-graining axioms
  * Showed the metric clusters representations by brain area (nearly blind to the encoder) and recovers synergy that Fisher information misses


### Equivariant Architectures for Visual Processing

* *Scale-Equivariant Networks for Neural Response Prediction* - Poster, *Dynamic Scale Equivariance in Retinal Neural Codes Supports Object Tracking*, CoSyNe 2026
  * Adapted Scale-Equivariant Steerable Networks for neural response prediction to naturalistic video stimuli
  * Achieved equivalent or superior performance using only 16k parameters, compared to conventional CNNs requiring over 100k parameters (84% reduction)
  * Showed scale equivariance is built into the retinal code in a cell-type-specific way (OFF-alpha ganglion cells)

* *Velocity-Equivariant Video Understanding* - In Progress
  * Developing a unified spatiotemporal equivariance framework that maintains parameter efficiency while enhancing performance on video understanding tasks
  * Validating preliminary results on dynamic stimuli

### Spatiotemporal Gaussian Processes for Predicting Retinal Neural Responses

* *Scalable gaussian process inference of neural responses to movies* [Accepted Poster at CoSyNe 2024](https://sazio.github.io/files/3D_GP_Cosyne2024.pdf) - Paper in Progress
  * Developed a scalable Gaussian process framework for modeling neural responses to naturalistic movie stimuli
  * Achieved results comparable with SOTA CNN models 

### Functional Cell-typing in large-scale electrophysiological recordings

* *Towards end-to-end cell-typing in large-scale electrophysiological recordings [Accepted Poster at CoSyNe 2024](https://sazio.github.io/files/MSF_Abstract_Cosyne2024.pdf)* - Paper in Progress
  * Created an end-to-end framework for automated identification of cell types in large-scale neural recordings
  * Leveraged functional response properties to classify different neuronal cell types

* *Estimating the Surround of Ganglion Cells in Large-Scale Recordings. [see Bernstein 2023 Poster](https://abstracts.g-node.org/conference/BC23/abstracts#/uuid/fba85980-a2a9-4c22-a584-979b2634eeab)*
  * Characterized center-surround receptive field properties in retinal ganglion cells
  * Developed methods for accurate estimation in large-scale multi-electrode array recordings


### Visual detection of elementary image features during natural behaviour 

* *Scene Structure Predicts Perceptual Decisions in Naturalistic Detection Tasks* (Yang, Vercillo, Cutrona, Azeglio, Iannetti, Neri) - bioRxiv 2026 (presented at ECVP 2025)
* Work led by Jun Yang at ENS, Paris. The project involved a VR experiment and training different DNNs; my contribution is on the deep-learning part.
*  Used deep neural networks to predict human perceptual decisions from trial-by-trial judgments, demonstrating that both local feature analysis and global scene context contribute equally to visual decision-making in naturalistic environments
