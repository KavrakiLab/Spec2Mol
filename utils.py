from rdkit import Chem
import numpy as np
import pickle

def check_smiles(smi):
    """ 
    check if a given smiles is valid
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return True

def canonicalise_smiles(smi):
    """
    return the canonical form of a given smiles 
    """
    mol = Chem.MolFromSmiles(smi)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonical

def get_species(mol):
    """
    Return the atom species that are present in a chemical molecule
    Args: rdkit molecule
    Output: set of atom species
    """
    species = set()
    for atom in mol.GetAtoms():
        species.add(atom.GetSymbol())
    return species

def spec2vec(spectrum,minmass,maxmass,resolution):
    """
    Convert a mass spectrum into a vector. Each bit in the vector corresponds to a discrete mass. 
    The value of each bit indicates the intensity for that mass.
    Args: 
        spectrum: set of tuples (mass,intensity)
        minmass: minimum allowed mass
        maxmass: maximum allowed mass 
        resolution: number of decimal points that are used to discretize the mass values (commonly 1 or 2)
    Output:
        vector representation of spectrum - normalized by dividing with the max mass.
    """
    mult = pow(10,resolution)
    length = int((maxmass-minmass)*mult)
    vec = np.zeros(length)
    if len(spectrum) == 0:
        return vec
    else:
        for (mass,abund) in spectrum:
            mass = round(float(mass),resolution)*mult
            try:
                vec[int(mass)-int(minmass*mult)] = vec[int(mass)-int(minmass*mult)] + abund
            except:
                print(mass)
        return vec/np.max(vec) 


def spec2tuples(spectrum):
    """
    convert a NIST spectrum into a set of tuples
    Args:
        spectrum as it is formatted in the NIST sdf files
    Output:
        set of tuples (mass, intensity)
    """
    tuples = set()

    for peak in spectrum.split('\n'):
        mass, abund, *c = peak.split(' ')
        mass = float(mass)
        abund = float(abund)
        tuples.add((mass,abund))
    return tuples


def spec2tuples_fromCSV(spectrum_file):
    """
    convert a spectrum stored in a csv file into a set of tuples
    Args:
        spectrum_file: spectrum csv file with the first column indicating the m/z ratio and the second column indicating the intensity  (comma separated columns)
    Output:
        set of tuples (mass, intensity)
    """
    tuples = set()
    entries = open(spectrum_file).read().split('\n')
    for i in range(0,len(entries)-1):
        entry = entries[i].split(',')
        mass = float(entry[0])
        abund = float(entry[1])
        tuples.add((mass,abund))
    return tuples

def remove_large(spectrum):
    remove = False
    for (mass,abund) in spectrum:
        mass = float(mass)
        if mass > 500:
            remove=True
    return remove


def writeout_spec(spectrum, outfile):
    writeout = open(outfile,'w')
    max_abund = 0
    for peak in spectrum.split('\n'):
        mass, abund, *c = peak.split(' ')
        if float(abund)>max_abund:
            max_abund = float(abund)
    max_abund = round(max_abund,2)
    vec = np.zeros(197500)
    for peak in spectrum.split('\n'):
        mass, abund, *c = peak.split(' ')
        abund = float(abund)
        rel_abund = abund/max_abund
        #mass = round(float(mass),2)*100
        #vec[int(mass)-2500] = rel_abund
        writeout.write(mass + ', ' + str(rel_abund) + '\n')
    writeout.close()
    return 

def min_max_mass(spectrum):
    """
    it finds the starting and the ending mass of a spectrum
    Args:
        A spectrum as it is formatted in the NIST sdf files
    Output:
        a tuple (starting mass, ending mass)
    """
    for peak in spectrum.split('\n'):
        mass, abund, *c = peak.split(' ')
        mass = round(float(mass),2)
        start = mass
        break
    for peak in spectrum.split('\n'):
        mass, abund, *c = peak.split(' ')
        mass = round(float(mass),2)
    end = mass
    return (start,end)

def create_valid(datafile, valid_size):
    validset = set()
    lines = open(datafile).read().split('\n')
    for smi in lines:
        if len(validset)<valid_size:
            validset.add(smi)
    return validset



def save_obj(obj, filename):
    """
    save an object in a pkl file
    """
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename):
    """
    load object from pkl file
    """
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


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

def read_smiles(filename):
    """
    read a txt file of line separated smiles into a set of smiles
    Args: filename: txt file directory
    Output: set of smiles
    """
    lines = open(filename).read().split('\n')
    smiles = set()
    for smi in lines:
        if check_smiles(smi):
            smiles.add(smi)
        else:
            print('Problem with SMILES: ', smi)
    return smiles 

def save_csv(directory, data):
    """
    save a numpy array into a csv file 
    Args:
        directory: outfile directory
        data: numpy array
    """
    with open(directory,'wb') as f:
        np.savetxt(f,data, fmt='%s', delimiter=',')
    return



def normalize_smiles(smi, canonical, isomeric):
    """
    for canonical smiles: canonocal=True and isomeric=False
    """
    normalized = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric)
    return normalized


def normalise_value(x, min_value, max_value):
    return (x-min_value)/(max_value-min_value)

def retrieve_value(x, min_value, max_value):
    return x*(max_value-min_value)+ min_value


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
