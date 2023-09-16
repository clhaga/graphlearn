import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import pandas as pd
from torch_geometric.data import Data

class MolecularGraphData:
    def __init__(self):
        self.permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn','Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au','Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    
    def one_hot_encoding(self, x, permitted_list):
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding

    def get_atom_features(self, atom):
        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), self.permitted_list_of_atoms)
        degree = self.one_hot_encoding(int(atom.GetDegree()), list(range(11)))
        valence = self.one_hot_encoding(int(atom.GetImplicitValence()), list(range(7)))
        formal_charge = [int(atom.GetFormalCharge())]
        radical_elec_enc = [int(atom.GetNumRadicalElectrons())]
        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["SP", "SP2", "SP3", "SP3D", "SP3D2"])
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4])
        is_in_a_ring_enc = [int(atom.IsInRing())]
        atom_feature_vector = atom_type_enc +  degree + valence + formal_charge + radical_elec_enc + hybridisation_type_enc + is_aromatic_enc + n_hydrogens_enc + is_in_a_ring_enc
        return np.array(atom_feature_vector)

    def get_bond_features(self, bond, use_stereochemistry=True):
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        return np.array(bond_feature_vector)

    def create_graph_data(self, x_smiles, y):  
        data_list = []
        for (smiles, y_val) in zip(x_smiles, y):
            mol = Chem.MolFromSmiles(smiles)
            n_nodes = mol.GetNumAtoms()
            n_edges = 2 * mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
            X = np.zeros((n_nodes, n_node_features))
            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = self.get_atom_features(atom)
            X = torch.tensor(X, dtype=torch.float)
            (rows, cols) = np.nonzero(Chem.AllChem.GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim=0)
            EF = np.zeros((n_edges, n_edge_features))
            for (k, (i, j)) in enumerate(zip(rows, cols)):
                EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
            EF = torch.tensor(EF, dtype=torch.float)
            y_tensor = torch.tensor([y_val], dtype=torch.long)  # Ensure y_val is a list or scalar
            data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
        return data_list
    
  
    def create_graph_solver_data(self, x_smiles):
        data_list = []
        for smiles in x_smiles:
            mol = Chem.MolFromSmiles(smiles)
            n_nodes = mol.GetNumAtoms()
            n_edges = 2 * mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
            X = np.zeros((n_nodes, n_node_features))
            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = self.get_atom_features(atom)
            X = torch.tensor(X, dtype=torch.float)
            (rows, cols) = np.nonzero(Chem.AllChem.GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim=0)
            EF = np.zeros((n_edges, n_edge_features))
            for (k, (i, j)) in enumerate(zip(rows, cols)):
                EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
            EF = torch.tensor(EF, dtype=torch.float)
            data_list.append(Data(x=X, edge_index=E, edge_attr=EF))
        return data_list




class SMILESToECFP4:
    def __init__(self):
        pass
    
    def smiles_to_ecfp4(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=2048)  # ECFP4 with 2048 bits
        arr = fp.ToBitString()
        arr = [int(bit) for bit in arr]
        return np.array(arr)
