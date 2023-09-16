import sys
sys.path.append('./utils')
sys.path.append('./graphs')
sys.path.append('./dataset')
import pandas as pd
import argparse
# import featurizer
from featurizer import MolecularGraphData, SMILESToECFP4
import graphalgos
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GeometricDataLoader




def main(args):
    device = torch.device('cuda')
    if args.model in ["GCN", "GIN", "GAT"]:
        df = pd.read_csv(args.csv_file)  
        x_smiles = df['Smiles'].tolist()
        data_list = MolecularGraphData().create_graph_solver_data(x_smiles)
        solve_loader = GeometricDataLoader(dataset = data_list, batch_size = args.batchsize, shuffle = False)
        predictions_list = []
        
        model = torch.load(args.model_path)
        model.eval()
        for data in solve_loader:
            with torch.no_grad():
                data.to(device)  
                _, output = model(data.x, data.edge_index, data.batch)
                predictions = output.cpu().detach().numpy().tolist()
                predictions_list.extend(predictions)
        
        result_df = pd.DataFrame({'Smiles': x_smiles, 'Predictions': predictions_list})    
        result_df.to_csv('GNN_predictions.csv', index=False)
    elif args.model == "DNN":
        data1 = pd.read_csv(args.csv_file)
        data1['X'] = data1['Smiles'].apply(SMILESToECFP4().smiles_to_ecfp4)
        data1 = data1[['Smiles','X']]
        X = torch.stack(data1['X'].apply(torch.from_numpy).tolist())
        predictions_list = []
        model = torch.load(args.model_path)
        model.eval()
        with torch.no_grad():
            outputs = model(X.to(device).float())
            predicted = torch.sigmoid(outputs)
            predictions = predicted.cpu().numpy()
                #predicted = torch.round(torch.sigmoid(outputs)).to(device)
                #predictions = predicted.cpu().detach().numpy().tolist()
                #print(predicted)
                #predictions_list.extend(predictions)
        data1['Prediction'] = predictions  
        data1.to_csv('DNN_predictions.csv', index=False)
    else:
        print("Invalid model selection. Available options: GCN, GIN, GAT, DNN")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning on Graphs.")

    parser.add_argument("--model", type=str, required = True, help="Deep learning model selection: Options - GCN, GIN, GAT, DNN")
    parser.add_argument("--model_path", type=str, required = True, help="Path to trained model")
    parser.add_argument("--batchsize", required = True, type=int, help="Batch size")
    parser.add_argument("--csv_file", required=True, type=str, help="Path to the Smiles CSV data file to process")
    args = parser.parse_args()
    main(args)