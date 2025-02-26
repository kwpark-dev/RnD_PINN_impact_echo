# RnD_PINN_impact_echo

## Overview
Impact-echo is one of the non-destructive testings to examine the quality of the product. It applies weak force on the target area (or we can say perturb the system) and gets response signals that are reflected from the system's boundary. If there's a crack inside the product, the response would contain distinctive modes. To train the model that performs binary signal classification, it demands a well-defined, enough amount of data. In case the dataset doesn't fulfill such conditions, signal synthesis is inevitable (due to time-consuming) and our approach can help the data collection problem.

## Objectives 
We suggest impact-echo signal generation via a physics-informed neural network (PINN). It directly solves partial differential equations using auto-gradient features offered from deep learning framework (pytorch or tensorflow). We want to address PDE solver using DNN step by step as follows.
1. 1D wave equation
2. 2D plate with perturbation
3. 3D pipe with perturbation and multi-physics feature

## Conclusion
Though the 1D problem was straightforward, the complexity beneath the PDE made the training harder. Error scales from different sources can screw up constraints such as initial/boundary conditions, perturbation, or even PDE violations. Such problems are addressed by a hard enforcement scheme, which ensures improvements of all errors.
