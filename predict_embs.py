import os
import sys
from os.path import exists
import argparse

import torch
import numpy as np

from spectra_encoder_model import Net1D
from utils import *


def main(args):

    minmass = 50.0  
    maxmass = 499.97
    resolution = 2

    model_dir = 'spectra_encoder_trained_model_200_epochs.pt'

    # pos_low_file = 'sample_data/[M+H]_low.csv'
    # pos_high_file = 'sample_data/[M+H]_high.csv'
    # neg_low_file = 'sample_data/[M-H]_low.csv'
    # neg_high_file = 'sample_data/[M-H]_high.csv'

    pos_low_file = args.pos_low_file
    pos_high_file = args.pos_high_file
    neg_low_file = args.neg_low_file
    neg_high_file = args.neg_high_file

    if pos_low_file is None and pos_high_file is None and neg_low_file is None and neg_high_file is None:
        sys.exit('No input spectra!')

    if not exists(pos_low_file) or pos_low_file is None:
        pos_low = set()
    else:
        pos_low = spec2tuples_fromCSV(pos_low_file)


    if not exists(pos_high_file) or pos_high_file is None:
        pos_high = set()
    else:
        pos_high = spec2tuples_fromCSV(pos_high_file)


    if not exists(neg_low_file) or neg_low_file is None:
        neg_low = set()
    else:
        neg_low = spec2tuples_fromCSV(neg_low_file)


    if not exists(neg_high_file) or neg_high_file is None:
        neg_high = set()
    else:
        neg_high = spec2tuples_fromCSV(neg_high_file)



    if remove_large(pos_low) or remove_large(pos_high) or remove_large(neg_low) or remove_large(neg_high):
        sys.exit('Invalid spectra: mass > 500')



    specvec_poslow = spec2vec(pos_low,minmass,maxmass,resolution).astype(np.float32)
    specvec_poshigh = spec2vec(pos_high,minmass,maxmass,resolution).astype(np.float32)
    specvec_neglow = spec2vec(neg_low,minmass,maxmass,resolution).astype(np.float32)
    specvec_neghigh = spec2vec(neg_high,minmass,maxmass,resolution).astype(np.float32)

    spectra = np.stack((specvec_poslow, specvec_poshigh, specvec_neglow, specvec_neghigh))

    spectra = torch.from_numpy(spectra)

    data_dim = spectra.shape[1]

    spectra = torch.unsqueeze(spectra,dim=0)

    model = Net1D(data_dim)
    model.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))

    model.eval()


    pred_emb = model(spectra)

    return pred_emb



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pos_low_file', type=str, default=None, help='csv file with positive mode [M+H]+ low energy spectrum')
    parser.add_argument('-pos_high_file', type=str, default=None, help='csv file with positive mode [M+H]+ high energy spectrum')
    parser.add_argument('-neg_low_file', type=str, default=None, help='csv file with positive mode [M-H]+ low energy spectrum')
    parser.add_argument('-neg_high_file', type=str, default=None, help='csv file with positive mode [M-H]- low energy spectrum')

'''
csv files format:
    1st column: m/z values
    2nd column: intensities
    comma separated
'''

    args = parser.parse_args()

    main(args)

