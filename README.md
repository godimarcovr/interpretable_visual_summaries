# Understanding Deep Architectures by Interpretable Visual Summaries [2/2]

### [M. Carletti](http://marcocarletti.altervista.org/), M. Godi, M. Aghaei, [M. Cristani](http://profs.sci.univr.it/~cristanm/)


Project @ [author's page](http://marcocarletti.altervista.org/publications/understanding-visual-summaries/)

Paper @ [https://arxiv.org/abs/1801.09103](https://arxiv.org/abs/1801.09103)

![visual summaries](http://marcocarletti.altervista.org/publications/understanding-visual-summaries/fig1.jpg)

---
**NOTE**

The project consists of two parts. Given a set of images belonging to the same class/category, the former part generates a crisp saliency mask for each image in the set. The second part computes a set of visual summaries starting from the crisp masks.

This is the SECOND part of the project.

You can find [HERE](https://github.com/mcarletti/interpretable-visual-summaries) the first part of the project concerning the computation of the crisp masks.

---

## Requirements
To generate crisp saliency maps (first part) you need to follow the instructions [here](https://github.com/mcarletti/interpretable-visual-summaries).

To generate a set of visual summaries (second part) for a specified class you need to:
* Install the MATLAB software (tested on Matlab2017b).
* Follow the installation instructions for [Proposal Flow](https://github.com/bsham/ProposalFlow) with [SelectiveSearch](http://koen.me/research/selectivesearch/) option as proposal method
* Set the path to ProposalFlow installation folder in 'set_proposal_flow_folder.m'
* Set 'in_base_folder' and 'out_base_folder' as, respectively, input and ouput folders in 'compute_clusters.m'; 'in_base_folder' should contain a folder for each class (each one produced by running the first part of the project) for which the summaries have to be computed.

## Usage [1/2]: generate crisp masks
Follow the instructions [here](https://github.com/mcarletti/interpretable-visual-summaries).

## Usage [2/2]: generate visual summaries
Run 'compute_clusters.m' to generate visual summaries. The resulting images are going to be separated into folders by class in 'out_base_folder', together with corresponding mat files that can be used to speed up future computations (to recompute summaries after changing parameters, remove the .mat files)