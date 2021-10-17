# Spec2Mol

Spec2Mol is a deep learning architecture for recommending molecular structures from MS/MS spectra.

Spec2Mol is an encoder-decoder architecture: The endoder creates an embedding from a given set of MS/MS spectra. The decoder reconstructs the molecular structure, in a SMILES format, given the embedding that the encoder generates.


The implementation of the Spec2Mol architecture is based on the Pytorch library.

Processing of the chemical data is based on the [RDKit](https://www.rdkit.org/) software.

## Installation

Create a conda environment:

```bash
conda create -n spec2mol python=3.7
source activate spec2mol
conda install rdkit -c rdkit
conda install pytorch=1.6.0 torchvision -c pytorch
```


