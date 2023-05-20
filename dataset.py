import os
import sys

import torch
import torch.utils.data

from utils import *

class MSDataset_4channels(torch.utils.data.Dataset):

    def __init__(self, partition, resolution, neg, augm=False, train_on_valid=False):
        """
        Args:
            partition (string): 'train', 'valid' or 'test'.
            resolution (int): number of decimal points. Choose either 1 or 2.
            neg (boolean): whether to use negative mode spectra
            augm (boolean): whether to augment the dataset
            train_on_valid (boolean): whether to use valid set in train (when applying model on test set)
        """
        path = os.getcwd()

        self.resolution = resolution
        self.neg = neg

        mass_thresh = 500
        self.minmass = 50.0  
        self.maxmass = 499.97
        if self.resolution == 1:
            self.maxmass = 500.1

        self.smiles_emb = torch.load(path + '/embeddings/smiles_NISTall_tanh.pt')    # dictionary mapping smiles to the embeddings learnt by the AE

        # dictionaries mapping smiles to spectra
        smiles_spec_MHposlow = load_obj('ms_data/data_4channels/[M+H]+/smiles2spectra_low')  # positive mode and low energy spectra
        smiles_spec_MHposhigh = load_obj('ms_data/data_4channels/[M+H]+/smiles2spectra_high')  # positive mode and high energy spectra

        smiles_spec_MHneglow = load_obj('ms_data/data_4channels/[M-H]-/smiles2spectra_low')  # negative mode and low energy spectra
        smiles_spec_MHneghigh = load_obj('ms_data/data_4channels/[M-H]-/smiles2spectra_high')  # negative mode and high energy spectra

        smiles_spec_MHposlow_augm = load_obj('ms_data/data_4channels_augm/[M+H]+/smiles2spectra_low') # augmented spectra - positive mode and low energy 
        smiles_spec_MHposhigh_augm = load_obj('ms_data/data_4channels_augm/[M+H]+/smiles2spectra_high')  # augmented spectra - positive mode and high energy 

        smiles_spec_MHneglow_augm = load_obj('ms_data/data_4channels_augm/[M-H]-/smiles2spectra_low')  # augmented spectra - negative mode and low energy 
        smiles_spec_MHneghigh_augm = load_obj('ms_data/data_4channels_augm/[M-H]-/smiles2spectra_high')  # augmented spectra - negative mode and high energy 
        

        smiles = read_smiles(path + '/4channel_data_updated/' + partition + '.txt')
        if partition == 'train' and train_on_valid:
            smiles = read_smiles(path + '/4channel_data_updated/train_valid.txt')


        self.id_smiles = {}
        self.id_poslow = {}
        self.id_poshigh = {}
        self.id_neglow = {}
        self.id_neghigh = {}
        i = 0
        for smi in smiles:
            remove = False
            self.id_smiles[i] = smi 
            if smi in smiles_spec_MHposlow.keys():
                pos_low = smiles_spec_MHposlow[smi]
            else:
                pos_low = set()
            if smi in smiles_spec_MHposhigh.keys():
                pos_high = smiles_spec_MHposhigh[smi]
            else:
                pos_high = set()
            if self.neg:
                if smi in smiles_spec_MHneglow.keys():
                    neg_low = smiles_spec_MHneglow[smi]
                else:
                    neg_low = set()
                if smi in smiles_spec_MHneghigh.keys():
                    neg_high = smiles_spec_MHneghigh[smi]
                else:
                    neg_high = set()
                for ms in neg_low:
                    if ms[0]>mass_thresh:
                        remove = True
                        continue
                for ms in neg_high:
                    if ms[0]>mass_thresh:
                        remove = True
                        continue
            if not remove:
                self.id_poslow[i] = pos_low
                self.id_poshigh[i] = pos_high
                if self.neg:
                    self.id_neglow[i] = neg_low
                    self.id_neghigh[i] = neg_high
                i = i + 1

        if partition == 'train' and augm:
            smiles_augm = read_smiles('ms_data/4channel_data_updated/train_augm.txt')  # smiles for which augmented spectra are available
            for smi in smiles_augm:
                remove = False
                if smi in smiles_spec_MHposlow_augm.keys():
                    pos_low = smiles_spec_MHposlow_augm[smi]
                elif smi in smiles_spec_MHposlow.keys():
                    pos_low = smiles_spec_MHposlow[smi]
                else:
                    pos_low = set()
                if smi in smiles_spec_MHposhigh_augm.keys():
                    pos_high = smiles_spec_MHposhigh_augm[smi]
                elif smi in smiles_spec_MHposhigh.keys():
                    pos_high = smiles_spec_MHposhigh[smi]
                else:
                    pos_high = set()
                for ms in pos_low:
                    if ms[0]>mass_thresh:
                        remove = True
                        continue
                for ms in pos_high:
                    if ms[0]>mass_thresh:
                        remove = True
                        continue
                if self.neg:
                    if smi in smiles_spec_MHneglow_augm.keys():
                        neg_low = smiles_spec_MHneglow_augm[smi]
                    elif smi in smiles_spec_MHneglow.keys():
                        neg_low = smiles_spec_MHneglow[smi]
                    else:
                        neg_low = set()
                    if smi in smiles_spec_MHneghigh_augm.keys():
                        neg_high = smiles_spec_MHneghigh_augm[smi]
                    elif smi in smiles_spec_MHneghigh.keys():
                        neg_high = smiles_spec_MHneghigh[smi]
                    else:
                        neg_high = set()
                    for ms in neg_low:
                        if ms[0]>mass_thresh:
                            remove = True
                            continue
                    for ms in neg_high:
                        if ms[0]>mass_thresh:
                            remove = True
                            continue
                if not remove:
                    self.id_smiles[i] = smi 
                    self.id_poslow[i] = pos_low
                    self.id_poshigh[i] = pos_high
                    if self.neg:
                        self.id_neglow[i] = neg_low
                        self.id_neghigh[i] = neg_high
                    i = i + 1


    def __len__(self):
        return len(self.id_smiles.keys())


    def __getitem__(self, idx):
        pos_low = self.id_poslow[idx]
        pos_high = self.id_poshigh[idx]
        specvec_poslow = spec2vec(pos_low,self.minmass,self.maxmass,self.resolution)
        specvec_poshigh = spec2vec(pos_high,self.minmass,self.maxmass,self.resolution)
        spectra = np.stack((specvec_poslow, specvec_poshigh))
        if self.neg:
            neg_low = self.id_neglow[idx]
            neg_high = self.id_neghigh[idx]
            specvec_neglow = spec2vec(neg_low,self.minmass,self.maxmass,self.resolution)
            specvec_neghigh = spec2vec(neg_high,self.minmass,self.maxmass,self.resolution)
            spectra = np.stack((specvec_poslow, specvec_poshigh, specvec_neglow, specvec_neghigh))
        spectra = torch.from_numpy(spectra)
        smiles = self.id_smiles[idx]
        smiles_norm = normalize_smiles(smiles, True, False)
        embedding = self.smiles_emb[smiles_norm]
        sample = {'spectra': spectra, 'embedding': embedding, 'smiles': smiles}
        return sample 


    def get_data_dim(self):
        idx = 0
        pos_low = self.id_poslow[idx]
        pos_high = self.id_poshigh[idx]
        specvec_poslow = spec2vec(pos_low,self.minmass,self.maxmass,self.resolution)
        specvec_poshigh = spec2vec(pos_high,self.minmass,self.maxmass,self.resolution)
        spectra = np.stack((specvec_poslow, specvec_poshigh))
        if self.neg:
            neg_low = self.id_neglow[idx]
            neg_high = self.id_neghigh[idx]
            specvec_neglow = spec2vec(neg_low,self.minmass,self.maxmass,self.resolution)
            specvec_neghigh = spec2vec(neg_high,self.minmass,self.maxmass,self.resolution)
            spectra = np.stack((specvec_poslow, specvec_poshigh, specvec_neglow, specvec_neghigh))
        spectra = torch.from_numpy(spectra)
        return spectra.shape
