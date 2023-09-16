# Deep Learning on Graphs

This repository contains code for deep learning on graphs using various graph neural network models. You can use this code to train and evaluate models such as GCN (Graph Convolutional Network), GIN (Graph Isomorphism Network), GAT (Graph Attention Network), and DNN (Deep Neural Network) on graph-structured data.  The repository consists of two programs: graphlearn.py and graphsolver.py.  Graphlearn.py is used to train a model.  Graphsolver.py is used to evaluate a trained model.  

## Getting Started

### Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- Python (>=3.8)
- PyTorch w/ cuda support (>=2.0)
- Pytorch Geometric (>=2.3)
- Pandas
- Argparse
- Torchsampler (0.1.2)
- Optuna (3.2.0)
- Scikit Learn (>=1.2.2)
- RDKit-pypi (2022.9.5)
- Numpy (>=1.23.5)
 

### Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/clhaga/graphlearn.git
   cd graphlearn

## Training a model with GraphLearn.py

The main script for training a neural network models is graphlearn.py. You can use various command-line arguments to specify the model, training parameters, and data:

Usage

   ```shell
   python graphlearn.py --model <model_name> --epochs <num_epochs> --batchsize <batch_size> --optimizer <optimizer_name> --          hidden_channels <num_hidden_channels> --learning_rate <learning_rate> --heads <num_heads> --optimization --optimization_cycles    <num_optimization_cycles> --csv_file <path_to_csv_data> [--imbalance] [--weight] [--sampler]
```
### Parameters
--model: Choose the deep learning model from available options: GCN, GIN, GAT, DNN. (required)

--epochs: Number of training epochs. (required)

--batchsize: Batch size for training. 

--optimizer: Choose the optimizer for training: Adam, RMSprop, SGD. 

--hidden_channels: Number of hidden channels (required for graph neural networks).

--learning_rate: Learning rate for optimization. 

--heads: Number of attention heads (requried for GAT model only).

--optimization: Run Optuna optimization for hyperparameter tuning.

--optimization_cycles: Number of Optuna optimization cycles to run.

--csv_file: Path to the CSV data file to process.

Imbalance Options:
You can enable imbalance handling with the following options:

--imbalance: Enable imbalanced dataset handling.

--weight: Calculate weights for imbalanced data handling.

--sampler: Use imbalanced data loader if data is imbalanced.

Note: When enabling imbalance handling (--imbalance), you must provide either --weight or --sampler. You cannot use both --weight and --sampler simultaneously with imbalance enabled.

The CSV file must have the format of Smiles in one column and Active in another column where Smiles is a SMILES structure of the compound and Active compounds are indicated by 1 and inactive compounds indicated by a 0.  

### Output

graphlearn.py outputs a trained model (either GNN_model.pt or DNN_model.pt) that can be used for evaluating compounds using graphsolver.py.

### Hyperparameter Tuning

graphlearn uses Optuna for hyperparameter tuning of batch size, learning rate, number of hidden channels, number of heads (for GAT), and optimizer.  To optimize a model, specify the model, number of epochs, and number of optimization cycles.

Usage

   ```shell
   python graphlearn.py --model <model name> --epochs <number of epochs> --optimization --optimization_cycles <number of optimization cycles>  --csv_file data.csv
```
This option outputs a CSV file of all training runs for manual inspection.  

## Evaluating Compounds with graphsolver.py

Usage

Graphsolver.py takes an input trained model from graphlearn.py and a CSV file containing SMILES (with a column header of Smiles).  It will then evaluate whether the compounds are active (1) or inactive (0).  The output is a CSV file with the SMILE and an array of (X,X) representing (0,1).  

   ```shell
   python graphsolver.py --model <model_name> --model_path <path_to_trained_model> --batchsize <batch_size> --csv_file <path_to_csv_data>
```
### Parameters

--model: Choose the deep learning model from available options: GCN, GIN, GAT, DNN.

--model_path: Path to the pre-trained model from graphlearn.py.

--batchsize: Batch size for making predictions (for GNN models).

--csv_file: Path to the CSV data file containing input data with "Smiles" as a column.

Example Usage

Make predictions using a pre-trained GIN model:
   ```shell
   python predict.py --model GIN --model_path GNN_model.pth --batchsize 64 --csv_file input_data.csv
```

# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for details.









