
# Struct-FV: A Graph-Based Approach for Spatially Coherent Feature Visualization

A Master's thesis research project conducted at the **Max Planck Institute for Informatics** and **Saarland University**, Germany.

**Researcher:** Swathi Suhas  
**Supervisors:** Prof. Dr. Bernt Schiele, Dr. Jonas Fischer, Ada Görgün

## Overview

Feature visualization is a post-hoc interpretability technique for understanding the internal representations learned by deep neural networks. While existing approaches improve the realism of generated visualizations through feature distribution matching, they often fail to preserve the spatial relationships between semantic features, resulting in distorted or structurally incoherent images.

**Struct-FV** addresses this limitation by introducing a graph-based feature visualization framework that explicitly models and preserves spatial relationships between important feature activations.

The method improves spatial coherence and interpretability while maintaining faithfulness to the underlying model, without relying on pretrained generative models or strong external priors.

## Key Contributions

- **Graph-Based Feature Representation:** Constructs graphs from salient spatial locations in feature maps, representing important features as nodes and their spatial relationships as edges.
- **Graph Matching:** Combines greedy feature-based node selection with Sinkhorn optimal transport to establish soft correspondences between synthetic and reference graphs.
- **Structural Loss:** Introduces a graph matching objective that preserves relative spatial relationships during feature visualization.
- **Reference Matching:** Uses random reference sampling and softmin aggregation to balance structural consistency, diversity, and computational efficiency.
- **Generalization:** Extends the framework to class-level, concept-level, and inner-neuron feature visualization.

## Methodology

Struct-FV extends feature distribution matching with an additional graph-based structural constraint.

The framework consists of four main stages:

1. **Graph Construction:** Select salient spatial locations from intermediate feature maps and construct fully connected graphs.
2. **Node Matching:** Identify corresponding locations between generated and reference images using feature similarity and greedy matching.
3. **Optimal Transport:** Apply Sinkhorn optimal transport to obtain soft node correspondences.
4. **Structural Optimization:** Minimize a graph matching loss based on pairwise spatial distances, encouraging coherent feature arrangements.

The resulting objective combines the VITAL feature distribution matching loss with the proposed structural loss:

\[
\mathcal{L}_{\text{Struct-FV}} =
\mathcal{L}_{\text{VITAL}} +
\lambda \mathcal{L}_{\text{graph}}
\]

## Experiments

Experiments were conducted on **ImageNet-1K** across four architectures:

- ResNet-50
- ConvNeXt-Base
- ViT-L/16
- ViT-L/32

Struct-FV was compared against existing feature visualization methods, including Fourier-based optimization, MACO, DeepInversion, and VITAL.

### Evaluation Metrics

- Fréchet Inception Distance (FID)
- Spatial FID (sFID)
- CLIP Maximum Mean Discrepancy (CMMD)
- CLIP Zero-Shot Classification Accuracy
- Original Model Classification Accuracy
- Qwen3-VL Semantic Evaluation
- Human Interpretability Study

## Results

### Qualitative Results

![Feature Visualization Comparison](assets/comparison.png)
Struct-FV improves semantic alignment and spatial coherence across CNN and Vision Transformer architectures while maintaining high model faithfulness.

### CLIP Zero-Shot Top-1 Accuracy (%)

| Architecture | VITAL | Struct-FV |
|---|---:|---:|
| ResNet-50 | 66.60 | **74.43** |
| ConvNeXt-Base | 63.40 | **65.70** |
| ViT-L/16 | 64.79 | **70.90** |
| ViT-L/32 | 55.97 | **67.70** |

Additional findings:

- Improved semantic alignment in external CLIP and Qwen3-VL evaluations across all four architectures.
- Improved or competitive image quality and spatial consistency compared with existing methods.
- A human study involving **29 participants** showed an overall preference for Struct-FV over VITAL in visual quality and interpretability.
- Demonstrated extensions to concept and inner-neuron visualization, with improvements depending on the spatial structure of the target features.

## Thesis

**Struct-FV: A Graph-Based Approach for Spatially Coherent Feature Visualization**

Master's Thesis, Saarland University, 2026.

[Read the Thesis](https://drive.google.com/file/d/1LbAmHmPiaqU4EuH9wvzqtPPX5gxcJ38B/view)
