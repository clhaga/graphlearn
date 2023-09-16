import sys
sys.path.append('./utils')
sys.path.append('./graphs')
sys.path.append('./dataset')
import pandas as pd
import argparse
# import featurizer
import build_dataset
import graphalgos
import splitters
import runtrain
import torch
import optimizer
from torch.utils.data import DataLoader
device = torch.device('cuda')

def main(args):
    if args.model in ["GCN", "GIN", "GAT"] and not args.optimization:
        trainer = runtrain.GNNModelTrainer(args.model, args.optimizer, args.learning_rate, args.epochs, args.sampler, args.weight, args.hidden_channels, args.heads, args.batchsize, args.csv_file)
        trainer.run()
    elif args.model in ["GCN", "GIN", "GAT"] and args.optimization:
        trainer = optimizer.OptimizeModelTrainer(args.model, args.epochs, args.optimization_cycles, args.weight, args.sampler, args.csv_file)
        trainer.run_optuna()
    elif args.model == "DNN":
        
        #model, train_loader, test_loader, optimizer, learning_rate, num_epochs, posweight
        trainer = runtrain.DNNModelTrainer(args.model, args.optimizer, args.learning_rate, args.epochs, args.sampler, args.weight, args.batchsize, args.csv_file)
        trainer.run()
    else:
        print("Invalid model selection. Available options: GCN, GIN, GAT, DNN")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning on Graphs.")

    parser.add_argument("--model", type=str, required = True, help="Deep learning model selection: Options - GCN, GIN, GAT, DNN")
    parser.add_argument("--epochs", type=int, help="Number of Epochs (Default 100)", default=20)
    parser.add_argument("--batchsize", type=int, help="Batch size")
    parser.add_argument('--imbalance', action='store_true', help='Enable imbalance handling')
    parser.add_argument("--optimizer", type=str, help="Optimizer selection: Adam, RMSprop, SGD (Default = Adam)", default='Adam')
    parser.add_argument("--hidden_channels", type=int, help="Number of hidden channels")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--heads", type=int, help="Heads: For GAT only")
    parser.add_argument("--optimization", action='store_true', help="Run Optuna optimization")
    parser.add_argument("--optimization_cycles", type=int, help="Number of Optuna optimization cycles to run")
    parser.add_argument("--csv_file", required=True, type=str, help="Path to the CSV data file to process")
    imbalance_group = parser.add_argument_group('Imbalance Options')
    imbalance_mutually_exclusive = imbalance_group.add_mutually_exclusive_group(required=False)
    imbalance_mutually_exclusive.add_argument('--weight', action='store_true', help='Calculate weights for imbalanced data handling')
    imbalance_mutually_exclusive.add_argument('--sampler', action='store_true', help='Use imbalanced loader if data imbalanced')

    args = parser.parse_args()
    if not args.optimization:
        if args.batchsize is None or args.learning_rate is None:
            parser.error("--batchsize and --learning_rate are required when not using optimization.")
    if args.imbalance:
        if not (args.weight or args.sampler):
            parser.error("When imbalance is enabled, you must provide either --weight or --sampler.")
        if args.weight and args.sampler:
            parser.error("You cannot use both --weight and --sampler at the same time with imbalance enabled.")
    if args.weight or args.sampler:
        if not args.imbalance:
            parser.error("When using --weight or --sampler, you must enable imbalance with --imbalance.")
    if args.model == "GAT" and not args.optimization:
        if args.heads is None:
            parser.error("When using GAT as the model, you must provide --heads.")
    if args.model != "DNN" and not args.optimization:
        if args.hidden_channels is None:
            parser.error("You must provide --hidden_channels for graph neural networks")
    if args.model == "DNN" and args.optimization:
        parser.error("Optimization has not been implemented for DNN")
    main(args)