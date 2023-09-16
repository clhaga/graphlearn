from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import torch
import pandas as pd
from featurizer import MolecularGraphData, SMILESToECFP4
from sklearn.model_selection import train_test_split


class MolecularDataset(InMemoryDataset):
    def __init__(self, root, csv_file, transform=None, pre_transform=None, pre_filter=None):
        self.csv_file = csv_file  
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.csv_file)  
        x_smiles = df['Smiles'].tolist()
        y = df['Active'].tolist()

        data_list = MolecularGraphData().create_graph_data(x_smiles, y)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




def process_dnn_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.rename(columns={'Active': 'label'})
    data = data.rename(columns={'Smiles': 'SMILES'})
    data['X'] = data['SMILES'].apply(SMILESToECFP4().smiles_to_ecfp4)
    data = data[['SMILES', 'X', 'label']]
    X = torch.stack(data['X'].apply(torch.from_numpy).tolist())
    y = torch.tensor(data['label'].values, dtype=torch.int16)
    return X, y


