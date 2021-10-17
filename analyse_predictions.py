import os
import numpy as np
import csv
from numpy import nan
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolDescriptors

from utils_eval import *
import operator
import torch



pubmed = True  # whether pubchem molecules are taken into account
only_pubmed = False   # only closest embeddings from the pretrained dataset are taken into account
visualize = False   # whether the predicted molecules will be visualizes

top_n = 20   # top molecules selected based on the MW criterion
topn_pubchem = 50   # number of closest pubchem molecules
filtering_MW = True  # filtering predictions based on molecular weight



tanimoto = []
cosine = []
avg_cosine = []
mcs_ratios = []
mcs_tans = []
mcs_coef = []
mcs_ratios_avg = []
mcs_tans_avg = []
mcs_coef_avg = []
valid = []
predlist_len = []
formulas = 0
exact = 0
correct_decodings = []
formula_dists = []
formula_dists_avg = []
heavy_atoms = []
formula_lengths = []
mw_diffs = []
mws = []
mws_avgs = []
pred_counts = []
overlap = 0
counter = 0


true_pos_atoms = {}
true_neg_atoms = {}
false_pos_atoms = {}
false_neg_atoms = {}
atom_list = set(['C', 'O', 'N', 'F', 'S', 'Cl', 'I', 'Br', 'P', 'Si', 'B'])
for at in atom_list:
    true_pos_atoms[at] = 0
    true_neg_atoms[at] = 0
    false_pos_atoms[at] = 0
    false_neg_atoms[at] = 0


model_decodings_file = 'ca17f425e_full_06_07_20beam.csv'

pubchem_decodings_file = 'topNmerged.pt'

figures_dir = 'Figures/'

original_topn = get_pubchem_mols(pubchem_decodings_file,topn_pubchem)

nospectra = 0 

idx = 0
with open(model_decodings_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        smiles = row['original_smiles']
        original = row['original_smiles']
        if pubmed:
            if not normalize_smiles(original, True, False) in original_topn.keys():
                continue
            pubmed_preds = original_topn[normalize_smiles(original, True, False)]
        original_mol = Chem.MolFromSmiles(original)
        formula_orig = rdMolDescriptors.CalcMolFormula(original_mol)
        formula_lengths.append(len(formula_orig))
        original_mw = Chem.Descriptors.ExactMolWt(original_mol)
        mws.append(original_mw)
        heavy_atoms.append(original_mol.GetNumHeavyAtoms())
        predicted = row['valid_smiles_list']
        pred_list = set(predicted.split('|'))
        if pubmed:
            for smi_pred in pubmed_preds:
                pred_list.add(smi_pred)
        if only_pubmed:
                pred_list = pubmed_preds
        pred_counts.append(len(pred_list))
        valid.append(len(get_valid(pred_list)))
        pred_list = get_valid(pred_list)
        if filtering_MW:
            pred_list, order_smiles = select_smiles_MW(pred_list, original_mw, top_n)
        predlist_len.append(len(pred_list))
        if visualize:
            vs = visualize_predictions(order_smiles, original, figures_dir, idx)
            idx = idx + 1
        smi1, tan = get_max_tanimoto(pred_list, original, 2, 32)
        smi2, cos = get_max_cosine(pred_list, original, 2, 32)
        smi_, cos_1024 = get_max_cosine(pred_list, original, 2, 1024)
        cos_avg = get_avg_cosine(pred_list, original, 2, 32)
        if cos_avg>0:
            avg_cosine.append(cos_avg)
        smi3, mcsr, mcs_tan, overlap_coef = get_max_mcs(pred_list,original)
        mcsr_avg, mcstan_avg, mcscoef_avg = get_avg_mcs(pred_list,original)
        correct_formula = compare_formulas(pred_list,original)
        if correct_formula:
            formulas = formulas + 1
        if cos_1024 == 1:
            exact = exact + 1
        form_dist = get_formulas_min_distance(pred_list,original)
        form_dist_avg = get_formulas_avg_distance(pred_list,original)
        formula_dists.append(form_dist)
        if form_dist_avg > 0:
            formula_dists_avg.append(form_dist_avg)
        tanimoto.append(tan)
        cosine.append(cos)
        mcs_ratios.append(mcsr)
        mcs_tans.append(mcs_tan)
        mcs_coef.append(overlap_coef)
        if  mcsr_avg>0 and mcstan_avg>0 and mcscoef_avg>0:
            mcs_ratios_avg.append(mcsr_avg)
            mcs_tans_avg.append(mcstan_avg)
            mcs_coef_avg.append(mcscoef_avg)
            mw_diff = get_MW_dif_min(pred_list, original)
        if mw_diff<100:
            mw_diffs.append(mw_diff)
            if get_MW_dif_avg(pred_list, original)>0:
                mws_avgs.append(get_MW_dif_avg(pred_list, original))
        for pred in pred_list:
             true_pos_atoms, true_neg_atoms, false_pos_atoms, false_neg_atoms = species_confusion(original, pred, true_pos_atoms, true_neg_atoms, false_pos_atoms, false_neg_atoms)

print("Total: ", len(tanimoto))

print("Overlap: ", overlap)

print("Cases removed - no spectra for sirius: ", nospectra)

print("Average number of valid predictions: ", np.mean(valid))

print("Average number of selected predictions: ", np.mean(predlist_len))


cnt = 0
for ii in range(0,len(valid)):
    if valid[ii]<10:
        cnt = cnt + 1

print("Number of cases with less than 10 predictions: ", cnt)


print("Exact structures found: ", exact)
print("Exact formulas found: ", formulas)

print("Average min MW difference: ", np.mean(mw_diffs))
print("Relative min MW difference: ", np.mean(mw_diffs)/np.mean(mws))

print("Average avg MW difference: ", np.mean(mws_avgs))
print("Relative avg MW difference: ", np.mean(mws_avgs)/np.mean(mws))


print("Average min formula distance: ", np.mean(formula_dists))
print("Relative min formula difference: ", np.mean(formula_dists)/np.mean(heavy_atoms))

print("Average avg formula distance: ", np.mean(formula_dists_avg))
print("Relative avg formula difference: ", np.mean(formula_dists_avg)/np.mean(heavy_atoms))


print("Average tanimoto: ", np.mean(tanimoto))

print("Average min cosine: ", np.mean(cosine))
print("Average avg cosine: ", np.mean(avg_cosine))

print("Average min mcs ratio: ", np.mean(mcs_ratios))
print("Average avg mcs ratio: ", np.mean(mcs_ratios_avg))

print("Average min mcs coef: ", np.mean(mcs_coef))
print("Average avg mcs coef: ", np.mean(mcs_coef_avg))

print("Average min mcs tans: ", np.mean(mcs_tans))
print("Average avg mcs tans: ", np.mean(mcs_tans_avg))


sensitivity = {}
specificity = {}
for atm in atom_list:
    try:
        sensitivity[atm] = true_pos_atoms[atm]/(true_pos_atoms[atm]+false_neg_atoms[atm])
        specificity[atm] = true_neg_atoms[atm]/(true_neg_atoms[atm]+false_pos_atoms[atm])
    except:
        print(atm)


print("Atom sensitivity: ", sensitivity)
print("Atom specificity: ", specificity)
