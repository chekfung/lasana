# LASANA: Large-Scale Surrogate Modeling for Analog Neuromorphic Architecture Exploration
**Authors:** Jason Ho, James. A. Boyle, Linshen Liu, and Andreas Gerstlauer \
**Affiliation:** The University of Texas at Austin, System-Level and Modeling Lab (SLAM Lab) \
**Conference:** Machine Learning on Computer-Aided Design (MLCAD) 2025 \
**Contact**: jason_ho@utexas.edu


### Abstract
Neuromorphic systems using in-memory or event-driven computing are motivated by the need for more energy-efficient processing of artificial intelligence workloads. Emerging neuromorphic architectures aim to combine traditional digital designs with the computational efficiency of analog computing and novel device technologies. A crucial problem in the rapid exploration and co-design of such architectures is the lack of tools for fast and accurate modeling and simulation. Typical mixed-signal design tools integrate a digital simulator with an analog solver like SPICE, which is prohibitively slow for large systems. By contrast, behavioral modeling of analog components is faster, but existing approaches are fixed to specific architectures with limited energy and performance modeling. In this paper, we propose LASANA, a novel approach that leverages machine learning to derive data-driven surrogate models of analog sub-blocks in a digital backend architecture. LASANA uses SPICE-level simulations of a circuit to train ML models that predict circuit energy, performance, and behavior at analog/digital interfaces. Such models can provide energy and performance annotation on top of existing behavioral models or function as replacements to analog simulation. We apply LASANA to an analog crossbar array and a spiking neuron circuit. Running MNIST and spiking MNIST, LASANA surrogates demonstrate up to three orders of magnitude speedup over SPICE, with energy, latency, and behavioral error less than 7%, 8%, and 2%, respectively.

[[arXiv]](https://arxiv.org/abs/2507.10748) [[Code Ocean]](optional-link)

---
### Overview
This repository contains the code and scripts used for our paper: LASANA: Large-Scale Surrogate Modeling for Analog Neuromorphic Architecture Exploration, accepted at MLCAD 2025. The code is structured for a Code Ocean capsule (link above), an immutable, reproducible container that contains code, datasets, and outputs from a run on their systems. We detail some of the limitations / guardrails put in place to successfully emulate our environment and results using Code Ocean below.

#### Limitations of Code Ocean
- For reproducability sake, much of the randomness associated with this code (random-generated inputs, ML model initial weights, etc.) have been set to a specific seed. In a #TODO: Future section, we will dictate exactly how to disable these guardrails to enable randomness. 

- Due to the limitations of Code Ocean capsules, any section that deals with proprietary CAD tools (testbench generation, dataset creation, SPICE runs, SystemVerilog Real Number Modeling Runs) have been commented out of the main script, but available to enable if one has access to these tools. 

- Another limitation for Code Ocean is the lack of runtime / space. Thus, a limited set of results are evaluated. Specifically, on the MNIST and Spiking MNIST runs, we only run the first 500 test images, rather than the full 10k image set used in the evaluation of LASANA in the paper. This can be changed by the user and detailed below, but requires access to CAD tools (Synopsys HSpice, and Cadence Spectre) to generate the full golden results. 

---
## A Brief Description of the Artifact
asdfasdf

### Claims and Experiments of the artifact

### Comparing Results in the Paper to the Artifact
asdfasdfadf








### Running Code Ocean Capsule for Artifact Analysis
1. The script should be configured to automatically run everything from the main script in the code repository, `run_lasana.py`. 
2. While intermediate results are created and not available in Code Ocean, the most important results are moved to the `/results/` directory, where they can be viewed.










## Using the Artifact for other things?
 - After training the models, how to use the actual models to do things and simulate things
 - Basically, I think we can just point them to the scripts for batch transient analysis
 - as well as the scripts for the mnist stuff?

 - We also provide all the scripts to generate everything.


### Disabling Reproducible Seed

### Enabling CAD tools
