# Deep Learning on Graphs

This repository contains code for deep learning on graphs using various graph neural network models. You can use this code to train and evaluate models such as GCN (Graph Convolutional Network), GIN (Graph Isomorphism Network), GAT (Graph Attention Network), and DNN (Deep Neural Network) on graph-structured data.  The repository consists of two programs: graphlearn.py and graphsolver.py.  graphlearn.py is used to train a model.  graphsolver.py is used to evaluate a trained model.  

## Getting Started

### Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- Python (>=3.8)
- PyTorch (>=2.0)
- Pytorch Geometric (>=2.3)
- Pandas
- Argparse
- Torchsampler
- Optuna
- Scikit Learn
- RDKit
- Numpy
- 

### Installation

1. Clone this repository:

   ```shell
   git clone https://github.com/clhaga/graphlearn.git
   cd graphlearn

### Training a model
Usage
The main script for training a neural network models is graphlearn.py. You can use various command-line arguments to specify the model, training parameters, and data:

   ```shell
   python graphlearn.py --model <model_name> --epochs <num_epochs> --batchsize <batch_size> --optimizer <optimizer_name> --          hidden_channels <num_hidden_channels> --learning_rate <learning_rate> --heads <num_heads> --optimization --optimization_cycles    <num_optimization_cycles> --csv_file <path_to_csv_data> [--imbalance] [--weight] [--sampler]
```
### Parameters
--model: Choose the deep learning model from available options: GCN, GIN, GAT, DNN.

--epochs: Number of training epochs.

--batchsize: Batch size for training.

--optimizer: Choose the optimizer for training: Adam, RMSprop, SGD.

--hidden_channels: Number of hidden channels (applicable for graph neural networks).

--learning_rate: Learning rate for optimization.

--heads: Number of attention heads (applicable for GAT model).

--optimization: Run Optuna optimization for hyperparameter tuning.

--optimization_cycles: Number of Optuna optimization cycles to run.

--csv_file: Path to the CSV data file to process.

Imbalance Options:
You can enable imbalance handling with the following options:

--imbalance: Enable imbalance handling.

--weight: Calculate weights for imbalanced data handling.

--sampler: Use imbalanced data loader if data is imbalanced.

Note: When enabling imbalance handling (--imbalance), you must provide either --weight or --sampler. You cannot use both --weight and --sampler simultaneously with imbalance enabled.

The CSV file must have the format of Smiles in one column and Active in another column where Smiles is a SMILES structure of the compound and Active compounds are indicated by 1 and inactive compounds indicated by a 0.  

## Hyperparameter Tuning

graphlearn uses Optuna for hyperparameter tuning of batch size, learning rate, number of hidden channels, number of heads (for GAT), and optimizer.  To optimize a model, specify the model, number of epochs, and number of optimization cycles.

   ```shell
   python graphlearn.py --model <model name> --epochs <number of epochs> --optimization --optimization_cycles <number of optimization cycles>  --csv_file data.csv
```
This option outputs a csv file of all training runs for manual inspection.  




