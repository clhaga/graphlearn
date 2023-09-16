import sys
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader as GeometricDataLoader, ImbalancedSampler
import build_dataset
import splitters
sys.path.append('../../graphs')
import graphalgos
class OptimizeModelTrainer:
    def __init__(self, model, num_epochs, num_cycles, posweight, sampler, csv_file):
        self.modeltype = model
        self.num_epochs = num_epochs
        self.use_sampler = sampler
        self.posweight = posweight
        self.num_cycles = num_cycles
        self.csv_file = csv_file
        


    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for data in self.train_loader: 
                data.to(self.device)
                _, out = self.model(data.x.float(), data.edge_index, data.batch) 
                loss = self.criterion(out, data.y).to(self.device)  
                loss.backward()  
                self.optimizer.step()  
                self.optimizer.zero_grad()   
                running_loss += loss.item()
                total_loss = running_loss/len(self.train_loader)
            #print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {running_loss/len(self.train_loader):.4f}")
            return total_loss

    def test(self):
        with torch.no_grad():
            self.model.eval()
            y_true = []
            y_pred = []
            correct = 0
            for data in self.test_loader:  # Iterate in batches over the training/test dataset.
                data.to(self.device)
                _, out = self.model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                y_true.append(data.y.cpu().numpy())
                y_pred.append(out.argmax(dim=1).cpu().numpy())
                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        
            y_true = torch.tensor(np.concatenate(y_true))
            y_pred = torch.tensor(np.concatenate(y_pred))
            accuracy = correct / len(self.test_loader.dataset)
            auc = roc_auc_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return auc, precision, recall, auprc, fn, fp, tp, tn, accuracy
            #print(f"AUC: {auc}, Precision: {precision}, Recall: {recall}, AUPRC: {auprc}, FN: {fn}, FP: {fp}, TP: {tp}, TN: {tn}, Accuracy: {accuracy}")
    
    def run_optuna(self):
        print("Preparing dataset...")    
        dataset = build_dataset.MolecularDataset(root='../', csv_file=self.csv_file)
        # Define the objective function for Optuna
        def objective(trial):
            self.device = torch.device('cuda')

            train_dataset, test_dataset = splitters.split_graphs(dataset)
            y = dataset._data.y
            class_weights= compute_class_weight(class_weight='balanced', classes= np.unique(y.numpy()), y = y.numpy())
            class_weights=torch.tensor(class_weights,dtype=torch.float)
            #weight = torch.tensor(self.posweight, dtype = torch.float32)
            if self.posweight: 
                self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to('cuda')
            else:
                self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
            hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256, 512])
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            if self.modeltype == "GAT":
                num_heads = trial.suggest_categorical("num_heads", [1, 2, 3, 4])
            else:
                pass 



            if self.modeltype == "GIN":
                self.model = graphalgos.GIN(num_node_features=dataset.num_node_features, dim_h = hidden_channels).to(self.device)
            elif self.modeltype == "GCN":
                self.model = graphalgos.GCN(num_node_features=dataset.num_node_features, num_classes = dataset.num_classes, dim_h = hidden_channels).to(self.device)
            elif self.modeltype == "GAT":
                self.model = graphalgos.GAT(num_node_features=dataset.num_node_features, num_classes = dataset.num_classes, dim_h = hidden_channels, num_heads=num_heads).to(self.device)
            


            # Create a new model and optimizer with the suggested hyperparameters
            model = self.model  # Replace with how you create your model
            

            if optimizer_name == 'Adam':
                self.optimizer = optim.Adam(model.parameters(), lr=lr)
            elif optimizer_name == 'RMSprop':
                self.optimizer = optim.RMSprop(model.parameters(), lr=lr)
            elif optimizer_name == 'SGD':
                self.optimizer = optim.SGD(model.parameters(), lr=lr)

            if self.use_sampler == True:
                print("Using imbalanced sampler")
                sampler = ImbalancedSampler(train_dataset)
                self.train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler= sampler)
                self.test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size,)
            else:
                self.train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                self.test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size,)

            self.num_epochs = self.num_epochs  # You may adjust this if needed

            # Train and test the model
            loss = self.train()
            auc, precision, recall, auprc, fn, fp, tp, tn, accuracy = self.test()

            
            return loss, auc, precision, recall, auprc, fn, fp, tp, tn, accuracy

        # Create an Optuna study and optimize the hyperparameters
        study = optuna.create_study(directions=["minimize", "maximize", "maximize", "maximize", "maximize", "minimize", "minimize", "maximize", "maximize", "maximize"])
        study.optimize(objective, n_trials=self.num_cycles)  # Adjust the number of trials as needed
        trials_df = study.trials_dataframe()
        trials_df.to_csv('./optimization_results.csv', index=False)
        # Print the best hyperparameters and results
        best_trial = max(study.best_trials)
        print(f"Best trial results: ")
        print(f"\tnumber: {best_trial.number}")
        print(f"\tparams: {best_trial.params}")
        print(f"\tvalues: Loss, AUC, Precision, Recall, AUPRC, FN, FP, TP, TN, Accuracy")
        print(f"\tvalues: {best_trial.values}")
        print("Manual selection of best trial may be necessary for imbalanced datasets")
        print("See optimization_results.csv for full trial results.")

        # Return the best model (you may need to adjust this part)
        return self.model

    def run(self):
        self.device = torch.device('cuda')
        weight = torch.tensor(self.posweight, dtype = torch.float32)
        self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
        # Train the model
        self.train()
        # Test the model
        self.test()


