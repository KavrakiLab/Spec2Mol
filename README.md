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

## Generate spectra embeddings:

```bash
python predict_embs.py -pos_low_file 'sample_data/[M+H]_low.csv' \
                     -pos_high_file 'sample_data/[M+H]_high.csv' \
                     -neg_low_file 'sample_data/[M-H]_low.csv' \
                     -neg_high_file 'sample_data/[M-H]_high.csv' \
```

where `pos_low_file`, `pos_high_file`, `neg_low_file`, `neg_high_file` are the csv files with the four input spectra:

`pos_low_file`: precursor [M+H]+, energy 35% NCE (Normalized Collision Energy)

`pos_high_file`: precursor [M+H]+, energy 130\% NCE

`neg_low_file`: precursor [M-H]-, energy 35\% NCE 

`neg_high_file`: precursor [M-H]-, energy 130\% NCE 


Each csv file has the m/z values in the first column and the intensity values in the second column.
The columns are separated with commas.
See file `sample_data`.

## Dataset
The spectra encoder has been trained on the [NIST Tandem Mass Spectral Library
2020](https://chemdata.nist.gov/dokuwiki/lib/exe/fetch.php?media=chemdata:asms2020:xiaoyu_yang_asms2020_presentation.pdf) which is a commercial dataset.


## Citation

```
@article{metatrans,
  author = {Litsa, Eleni E. and Chenthamarakshan, Vijil and Das, Payel and Kavraki, Lydia E.},
  title = {An end-to-end deep learning framework for translating mass spectra to de-novo molecules},
  journal = {Communications Chemistry},
  year = {2023},
}
```