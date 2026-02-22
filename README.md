# APIC-Net
**Author**: Utkarsh Tyagi

## Overview
**APIC-Net** is a research project aimed at building a cyberattack detection system for Virtual Synchronous Machines (VSMs) in power grids, specifically tested on the IEEE 39-bus (New England) system.

This repository implements the **DATASET** pipeline, responsible for simulating normal operation and various cyberattack scenarios (FDI attacks) against VSMs within the power grid.

## Dataset Pipeline

The APIC-Net Data Generation Pipeline leverages `pandapower` along with custom Euler integration simulations to produce raw, physics-informed data for testing anomaly detection architectures.

### Simulated System
* **Grid Topography**: Standard IEEE 39-bus (New England) system.
* **Nodes**: 10 generators, where 6 have been replaced with Virtual Synchronous Machines (VSMs) to enable meaningful federated learning applications.
* **Dynamics**: Incorporates nonlinear swing dynamics which are mathematically vital for evaluating cross-physics invariant networks.

### Dataset Structure
The generated datasets are automatically isolated per VSM (VSM0 through VSM5) and partitioned into normal and attack scenarios. The following attack types are generated:
* `freq`: Frequency sensor spoofing
* `coi`: Center of Inertia (CoI) spoofing
* `power`: Active power measurement spoofing
* `voltage`: Voltage reference tampering

**Location**: All generated raw arrays and tracking metadata JSONs are stored natively within the `data/` directory.

### Usage
The pipeline primarily runs through:
```bash
python -m simulation.generate_dataset
```
This handles the topology construction, physical differential equations, noise injections, and dataset splitting into PyTorch-consumable `.npz` files.
