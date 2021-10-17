import numpy as np
from numpy import nan
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdMolDescriptors

def read_smiles(filename):
    """
    read a txt file of line separated smiles into a set of smiles
    Args:
        filename: txt file directory
    Output: 
        set of smiles 
    """
    lines = open(filename).read().split('\n')
    smiles = set()
    for smi in lines:
        smiles.add(smi)
    return smiles


def writeout_smiles(filename, smiles_set):
    """
    writes a set of smiles into a txt file
    Args: 
        filename: output txt file directory
        smiles_set: set of smiles 
    """
    outfile = open(filename,'w')
    for smiles in smiles_set:
        outfile.write(smiles + '\n')
    outfile.close()
    return

def load_obj(filename):
    """
    load a pkl object
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def check_smiles(smi):
    """ 
    check smiles for validity
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return True


def normalize_smiles(smi, canonical, isomeric):
    """
    for canonical smiles: canonical =True and isomeric=False 
    """
    normalized = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric)
    return normalized

def get_valid(pred_list):
    valid = set()
    for smi in pred_list:
        try:
            check = check_smiles(smi)
            if check:
                valid.add(smi)
        except:
            removed = removed + 1
    return valid


def get_species(smiles):
    """
    Return the atom species that are present in a chemical molecule
    Args: smiles representation
    Output: set of atom species
    """
    species = set()
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        for at in mol.GetAtoms():
            species.add(at.GetSymbol())
    return species            


def get_tanimoto(smiles1, smiles2, radius, bits):
    """
    tanimoto fingerprint similarity between 2 smiles
    Args:
        smiles1, smiles2 to be compared 
        radius and bits for the fingerprint representation
    Output: 
        tanimoto similarity
    """
    m1 = Chem.MolFromSmiles(smiles1)
    m2 = Chem.MolFromSmiles(smiles2)
    if m1 and m2:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,radius,nBits=bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2,radius,nBits=bits)
        return DataStructs.FingerprintSimilarity(fp1, fp2)
    else:
        return 0


def get_cosine(smiles1, smiles2, radius, bits):
    """
    cosine fingerprint similarity between 2 smiles
    Args: 
        smiles1, smiles2 to be compared 
        radius and bits for the fingerprint representation
    Output:
        cosine similarity 
    """
    m1 = Chem.MolFromSmiles(smiles1)
    m2 = Chem.MolFromSmiles(smiles2)
    if m1 and m2:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,radius,nBits=bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2,radius,nBits=bits)
        return DataStructs.CosineSimilarity(fp1, fp2)
    else:
        return 0


def get_RDFsim(smiles1, smiles2):
    """
    Fingerprint similarity based on RDKFingerprint
    """
    original_fgp = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles1))
    predicted_fgp = Chem.RDKFingerprint(Chem.MolFromSmiles(smiles2))
    sim = DataStructs.FingerprintSimilarity(original_fgp,predicted_fgp)
    return sim


def get_max_tanimoto(pred_list, original, radius, bits):
    """
    find the predicted smiles with the maximum tanimoto similarity with respect to the original smiles 
    Args:
        pred_list: list of predicted smiles 
        original: reference smiles 
        radius and bits for the fingerprint representation
    Output:
        closst_smiles: smiles with the max tanimoto 
        max_tan: maximum tanimoto 

    """
    max_tan = 0
    closest_smiles = ''
    for smi in pred_list:
        tan = get_tanimoto(original, smi, radius, bits)
        if tan > max_tan:
            max_tan = tan
            closest_smiles = smi
    return closest_smiles, max_tan


def get_max_cosine(pred_list, original, radius, bits):
    """
    find the predicted smiles with the maximum cosine similarity 
    Args:
        pred_list: list of predicted smiles 
        original: reference smiles 
        radius and bits for the fingerprint representation
    Output:
        closest_smiles: smiles with the max cosine similarity 
        max_cos: maximum cosine 
    """
    max_cos = 0
    closest_smiles = ''
    for smi in pred_list:
        cos = get_cosine(original, smi, radius, bits)
        if cos > max_cos:
            max_cos = cos
            closest_smiles = smi
    return closest_smiles, max_cos

def get_avg_cosine(pred_list, original, radius, bits):
    """
    average cosine fingerprint similarity among all predictions
    Args:
        pred_list: list of predicted smiles 
        original: reference smiles 
        radius and bits for the fingerprint representation
    Output:
        average cosine similarity 
    """
    coss = []
    for smi in pred_list:
        cos = get_cosine(original, smi, radius, bits)
        if cos>0:
            coss.append(cos)
    return np.mean(coss)



def get_max_mcs(pred_list,original):
    """
    MCS-based metrics of the prediction with the max number of atoms in the maximum common substructure
    Args:
        pred_list: list of predicted smiles
        original: reference smiles 
    Output:
        closest_smiles: smiles with the maximum number of atoms in the MCS 
        mcs ratio for the smiles with the max number of atoms in the MCS 
        mcs tanimoto >>
        mcs coefficient  >>
    """
    removed = 0
    original_mol = Chem.MolFromSmiles(original)
    original_atoms = original_mol.GetNumAtoms()
    max_mcs = 0
    mcs_atoms = 0
    closest_smiles = ''
    for smi in pred_list:
        try:
            mols = [Chem.MolFromSmiles(original), Chem.MolFromSmiles(smi)]
            mcs = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True, atomCompare=Chem.rdFMCS.AtomCompare.CompareElements, bondCompare=Chem.rdFMCS.BondCompare.CompareOrder, timeout=60)
            mcs_atoms = mcs.numAtoms
            if mcs_atoms>max_mcs:
                max_mcs = mcs_atoms
                closest_smiles = smi
        except:
            removed = removed + 1
    closest_mol = Chem.MolFromSmiles(closest_smiles)
    closest_atoms = closest_mol.GetNumAtoms()
    if mcs_atoms == 0:
        mcs_ratio = 0
        mcs_tan = 0
        mcs_coef = 0
    else:
        mcs_ratio = max_mcs/original_atoms
        mcs_tan = max_mcs/(original_atoms+closest_atoms-max_mcs)
        mcs_coef = max_mcs/min(original_atoms, closest_atoms)
    return closest_smiles, mcs_ratio, mcs_tan, mcs_coef


def get_avg_mcs(pred_list,original):
    """
    Average values of the MCS-based metrics between the predicted smiles and the reference smiles
    Args: 
        pred_list: list of predicted smiles
        original: reference smiles
    Output: 
    average mcs ratio 
    average mcs tanimoto 
    average mcs coefficient
    """
    removed = 0
    original_mol = Chem.MolFromSmiles(original)
    original_atoms = original_mol.GetNumAtoms()
    mcs_rs = []
    mcs_tns = []
    mcs_cfs = []
    for smi in pred_list:
        try:
            pred_mol = Chem.MolFromSmiles(smi)
            pred_atoms = pred_mol.GetNumAtoms()
            mols = [original_mol,pred_mol]
            mcs = rdFMCS.FindMCS(mols, ringMatchesRingOnly=True, atomCompare=Chem.rdFMCS.AtomCompare.CompareElements, bondCompare=Chem.rdFMCS.BondCompare.CompareOrder, timeout=60)
            mcs_atoms = mcs.numAtoms
            if mcs_atoms == 0:
                mcs_ratio = 0
                mcs_tan = 0
                mcs_coef = 0
            else:
                mcs_ratio = mcs_atoms/original_atoms
                mcs_tan = mcs_atoms/(original_atoms+pred_atoms-mcs_atoms)
                mcs_coef = mcs_atoms/min(original_atoms, pred_atoms)
                mcs_rs.append(mcs_ratio)
                mcs_tns.append(mcs_tan)
                mcs_cfs.append(mcs_coef)
        except:
            removed = removed + 1
    return np.mean(mcs_rs), np.mean(mcs_tns), np.mean(mcs_cfs)    



def compare_formulas(pred_list,original):
    """
    check whether any of the predicted smiles has the correct molecular formula (with respect to the reference molecule)
    Args: 
        pred_list: list of predicted smiles 
        original: reference smiles
    Output:
        True/False 
    """
    removed = 0
    original_mol = Chem.MolFromSmiles(original)
    formula_orig = rdMolDescriptors.CalcMolFormula(original_mol)
    pred_formulas = set()
    for smi in pred_list:
        try:
            predicted_mol = Chem.MolFromSmiles(smi)
            formula = rdMolDescriptors.CalcMolFormula(predicted_mol)
            pred_formulas.add(formula)
        except:
            removed = removed + 1
    if formula_orig in pred_formulas:
        return True
    else:
        return False


def get_formulas(formulas_file):
    formulas = set()
    lines = open(formulas_file).read().split('\n')
    for line in lines[1:]:
        if not line == '':
            line = line.split('\t')
            formulas.add(line[1])
    return formulas

def get_structures(structures_file, topn):
    structures = set()
    lines = open(structures_file).read().split('\n')
    for line in lines[1:]:
        if len(structures) == topn:
            break
        if not line == '':
            line = line.split('\t')
            structures.add(normalize_smiles(line[8], True, False))
    return structures



def get_formulas_min_distance(pred_list, original, hydrogens=False):
    """
    Get minimum distance between predicted molecular formulas and the reference molecular formula.
    Distance is defined as the number of atoms that differ between two molecules 
    Args:
        List of predicted smiles
        original: reference smiles
        hydrogens: if False then hydrogens are not taken into account
    Output:
        minimum distance over all distances from each predicted smiles 
    """
    min_distance = 100
    removed = 0
    for smi in pred_list:
        try:
            dist = compare_species_counts(smi,original,hydrogens)
            if dist<min_distance:
                min_distance = dist 
        except:
            removed = removed + 1
    return min_distance


def get_formulas_avg_distance(pred_list, original, hydrogens=False):
    """
    Get the average over all distances between the molecular formulas of the predicted smiles and the molecular formula of the reference smiles
    Distance is defined as the number of atoms that differ between the two molecules
    Args:
        pred_list: list of predicted smiles
        original: reference smiles 
        hydrogens: if False then hydrogens are not taken into account
    Output:
        average distance over all distances from each predicted smiles 
    """
    removed = 0 
    dists = []
    for smi in pred_list:
        try:
            dist = compare_species_counts(smi, original, hydrogens)
            dists.append(dist)
        except:
            removed = removed + 1
    return np.mean(dists)


def compare_species(smi_ref, smi_pred):
    """
    compares the atom species between a predicted smiles and a reference smiles 
    without accounting for discrepancies for the number of atoms for each species
    Args:
        smi_ref: reference smiles 
        smi_pred: predicted smiles 
    Output:
        found: set of atom species that have been found in the predicted smiles and the reference smiles 
        not_found: set of atom species that are found in the reference smiles but not i the predicted smiles
    """
    found = set()
    not_found = set()
    species_ref = get_species(smi_ref)
    species_pred = get_species(smi_pred)
    for spe in species_ref:
        if spe in species_pred:
            found.add(spe)
        else:
            not_found.add(spe)
    return found, not_found


def species_dif(found,not_found, atom_counts_found, atom_counts_notfound):
    """
    It aggregates the found/not found atom species in the entire dataset
    Args:
    found/not_found: set of atom species found/not found for a single molecule 
    atom_counts_found/atom_counts_notfound: dictionaries that keep track of the number of molecules 
    for which a specific tom species from the reference smiles has been / has not been found in the
    predicted smiles 
    """
    for spe in found:
        atom_counts_found[spe] = atom_counts_found[spe] + 1 
    for spe in not_found:
        atom_counts_notfound[spe] = atom_counts_notfound[spe] + 1 
    return atom_counts_found, atom_counts_notfound


def count_hydrogens(smiles):
    """
    counts the number of hydrogens in a given smiles 
    """
    hydrogens = 0
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)
    heavy_atoms = mol.GetNumAtoms()
    molH = Chem.AddHs(mol)
    all_atoms = molH.GetNumAtoms()
    if all_atoms - heavy_atoms > 0:
        hydrogens = all_atoms - heavy_atoms
    return hydrogens


def get_species_counts(smiles, hydrogens=False):
    """
    creates a vocabulary with the number of atoms for each atom species
    """
    species_counts = {}
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        species = atom.GetSymbol()
        if species in species_counts.keys():
            counts = species_counts[species]
        else:
            counts = 0
        counts = counts + 1
        species_counts[species] = counts
    if hydrogens:
        species_counts['H'] = count_hydrogens(smiles)
    return species_counts



def compare_species_counts(smiles_1, smiles_2, hydrogens=False):
    """
    computes the molecular formula distance:
    """
    species_counts_1 = get_species_counts(smiles_1)
    species_counts_2 = get_species_counts(smiles_2)
    errors = 0
    for spe in species_counts_1.keys():
        counts1 = species_counts_1[spe]
        if spe in species_counts_2.keys():
            counts2 = species_counts_2[spe]
        else:
            counts2 = 0
        errors = errors + abs(counts1-counts2)
    for spe in species_counts_2.keys():
        if not spe in species_counts_1.keys():
            errors = errors + species_counts_2[spe]
    return errors


def get_MW_dif_min(pred_list, original):
    """
    finds the minimum deviation from the reference molecular weight among all predicted smiles
    """
    original_mol = Chem.MolFromSmiles(original)
    original_mw = Chem.Descriptors.ExactMolWt(original_mol)
    min_diff = 100
    rem = 0
    for pred in pred_list:
        try: 
            predicted_mol = Chem.MolFromSmiles(pred)
            predicted_mw = Chem.Descriptors.ExactMolWt(predicted_mol)
            diff = abs(original_mw-predicted_mw)
            if diff < min_diff:
                min_diff = diff
        except: 
            rem = rem + 1
    return min_diff

def species_confusion(smi_ref, smi_pred, true_pos, true_neg, false_pos, false_neg):
    """
    finds true positives, true negatives, false positives and false negatives in the detection
    of the atom species in the predicted smiles without taking into account the atom counts per species 
    Args: 
        smi_ref: reference smiles
        smi_pred: predicted smiles 
        true_pos: dictionary that keeps track of the number of true positives (at a molecule level) per species 
        true_neg: dictionary...
        false_pos: dictionary... 
        false_neg: ...
    Output 
        Updated dictionaries 
    """
    species_ref = get_species(smi_ref)
    species_pred = get_species(smi_pred)
    if not len(species_pred)==0:
        for spe in species_ref:
            if spe in true_pos.keys():
                if spe in species_pred:
                    true_pos[spe] = true_pos[spe] + 1
        for spe in species_ref:
            if spe in true_pos.keys():
                if not spe in species_pred:
                    false_neg[spe] = false_neg[spe] + 1
        for spe in species_pred:
            if spe in true_pos.keys():
                if not spe in species_ref:
                    false_pos[spe] = false_pos[spe] + 1
        for spe in true_pos.keys():
            if not spe in species_ref and not spe in species_pred:
                true_neg[spe] = true_neg[spe] + 1
    return true_pos, true_neg, false_pos, false_neg
    


def get_MW_dif_avg(pred_list, original):
    """
    finds the average deviation from the reference molecular weight among all predicted smiles 
    """
    original_mol = Chem.MolFromSmiles(original)
    original_mw = Chem.Descriptors.ExactMolWt(original_mol)
    dmws = []
    rem = 0
    for pred in pred_list:
        try: 
            predicted_mol = Chem.MolFromSmiles(pred)
            predicted_mw = Chem.Descriptors.ExactMolWt(predicted_mol)
            diff = abs(original_mw-predicted_mw)
            dmws.append(diff)
        except: 
            rem = rem + 1
    return np.mean(dmws)



def get_pubchem_mols(infile, topn):
    pubmols = torch.load(infile)
    original_topn = {}
    for smi in pubmols:
        smiles_dist = {}
        topnset = set()
        dist_smiles = pubmols[smi]
        for ii in range(0,len(dist_smiles)):
            dist = dist_smiles[ii][0]
            smiles = dist_smiles[ii][1]
            smiles_dist[smiles] = dist
        sorted_smiles = sorted(smiles_dist.items(), key=operator.itemgetter(1))
        ii = 0
        while len(topnset)<topn and ii<len(sorted_smiles):
            topnset.add(sorted_smiles[ii][0])
            ii = ii + 1
        original_topn[normalize_smiles(smi, True, False)] = topnset
    return original_topn


def select_smiles_MW(pred_list, original_mw, top_n):
    '''
    ranks predictions based on the deviation from the original MW
    and returns the topn
    Args:
        pred_list: list of predicted smiles
        original_mw: MW of the molecule
        top_n: number of selected molecules
    '''
    smiles_MWdiff = {}
    rem = 0
    selected = set()
    order_smiles = {}
    for smi in pred_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            mw = Chem.Descriptors.ExactMolWt(mol)
            diff = abs(original_mw-mw)
            smiles_MWdiff[smi] = diff
        except:
            rem = rem + 1
    sorted_smiles = sorted(smiles_MWdiff.items(), key=operator.itemgetter(1))
    ii = 0
    while len(selected)<top_n and ii<len(sorted_smiles):
        selected.add(sorted_smiles[ii][0])
        order_smiles[ii] = sorted_smiles[ii][0]
        ii = ii + 1
    return selected, order_smiles


def visualize_predictions(order_smiles, original, figures_dir, idx):
    '''
    Visualizes predictions and the reference molecule
    Args:
        order_smiles: dictionary with the top-n ordered smiles (dictionary maps the orderring indices to the smiles)
        original: reference molecule in smiles
        figures_dir: directory to save figures
        idx: id number of the molecule 
    '''
    work_dir = figures_dir + 'mol_' + str(idx) + '/'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    filename = work_dir + 'original.png'
    mol = Chem.MolFromSmiles(original)
    Draw.MolToFile(mol,filename)
    for ii in order_smiles.keys():
        filename = work_dir + str(ii) + '.png'
        mol = Chem.MolFromSmiles(order_smiles[ii])
        Draw.MolToFile(mol,filename)
    return 1


