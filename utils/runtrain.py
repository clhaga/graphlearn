import sys
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score, confusion_matrix
import optuna
import build_dataset
import splitters
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.loader import DataLoader as GeometricDataLoader, ImbalancedSampler
sys.path.append('../../graphs')
sys.path.append('../../utils')
import graphmask_explainer
from torch_geometric.explain import Explainer
import graphalgos
from torchsampler import ImbalancedDatasetSampler
from sklearn.utils.class_weight import compute_class_weight


class DNNModelTrainer:
    #args.model, args.optimizer, args.learning_rate, args.epochs, args.sampler, args.weight, args.batchsize, args.csv_file
    def __init__(self, model, optimizer, learning_rate, epochs, sampler, weight, batchsize, csv_file):
        self.model = model
        self.learning_rate = learning_rate
        self.use_sampler = sampler
        self.batchsize = batchsize
        self.csv_file = csv_file
        self.num_epochs = epochs
        self.posweight = weight
        self.optimizer = optimizer
   
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device).float()
                labels = labels.unsqueeze(1).float().to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {running_loss/len(self.train_loader):.4f}")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device).float()
                labels = labels.unsqueeze(1).float().to(self.device)
                outputs = self.model(inputs)
                predicted = torch.round(torch.sigmoid(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.append(labels.float())
                y_pred.append(predicted.float())
            accuracy = correct / total
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)
            num_classes = len(torch.unique(y_true))
            confusion = torch.zeros(num_classes, num_classes, device=self.device)
            indices = y_true * num_classes + y_pred
            indices = indices.view(-1).long()
            confusion = confusion.view(-1).index_add_(0, indices, torch.ones_like(indices, device=self.device).float())
            confusion = confusion.view(num_classes, num_classes)
            confusion = confusion.cpu()
            tn, fp, fn, tp = confusion.numpy().ravel()
            print(f"Test Accuracy: {accuracy:.4f}, TN: {tn:.4f}, FP: {fp:.4f}, FN: {fn:.4f}, TP: {tp:.4f}")

    def run(self):
        self.device = torch.device('cuda')
        model_save_path = 'DNN_model.pth'
        print("Selected DNN model")
        print("Preparing dataset...")
        X,y = build_dataset.process_dnn_data(self.csv_file)
        train_dataset, test_dataset  = splitters.split_data(X,y)
        self.model = graphalgos.DNN().to(self.device)
        if self.use_sampler:
            self.train_loader = DataLoader(train_dataset, shuffle=False, batch_size=self.batchsize, sampler=ImbalancedDatasetSampler(train_dataset))
            self.test_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batchsize)
        else:     
            self.train_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batchsize)
            self.test_loader = DataLoader(test_dataset, batch_size=self.batchsize)
        #select optimizer
        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.posweight:
            pass
            #class_weight=torch.tensor(class_weights,dtype=torch.float)
       
        if self.posweight: 
            self.criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
        
        # Train the model
        self.train()
        # Test the model
        self.test()
        torch.save(self.model, model_save_path)
        print(f"Model saved to {model_save_path}")






class GNNModelTrainer:
    def __init__(self, modeltype, optimizer, learning_rate, num_epochs, sampler, posweight, hidden_channels, heads, batchsize, csv_file):
        self.modeltype = modeltype
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_sampler = sampler
        self.posweight = posweight
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.optimizer = optimizer
        self.batchsize = batchsize
        self.csv_file = csv_file

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for data in self.train_loader: 
                data.to(self.device)
                _, out = self.model(data.x.float(), data.edge_index, data.batch) 
                loss = self.criterion(out, data.y).to(self.device)
                  
                # out = self.model(data.x.float(), data.edge_index, data.batch)
                # loss = self.criterion(torch.squeeze(out), data.y.float()).to(self.device)  
                
                loss.backward()  
                self.optimizer.step()  
                self.optimizer.zero_grad()   
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {running_loss/len(self.train_loader):.4f}")

    def test(self):
        self.model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []
            correct = 0
            for data in self.test_loader:  # Iterate in batches over the training/test dataset.
                data.to(self.device)
                _, out = self.model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.
                # out = self.model(data.x, data.edge_index, data.batch)
                # pred = (torch.sigmoid(out) > 0.5).int()
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
            print(f"AUC: {auc}, Precision: {precision}, Recall: {recall}, AUPRC: {auprc}, FN: {fn}, FP: {fp}, TP: {tp}, TN: {tn}, Accuracy: {accuracy}")


    def run(self):
        model_save_path = 'GNN_model.pth'
        self.device = torch.device('cuda')
        print("Preparing dataset")    
        dataset = build_dataset.MolecularDataset(root='../', csv_file=self.csv_file)
        train_dataset, test_dataset = splitters.split_graphs(dataset)
        if self.use_sampler == True:
            print("Using imbalanced sampler")
            sampler = ImbalancedSampler(train_dataset)
            self.train_loader = GeometricDataLoader(train_dataset, batch_size=self.batchsize, shuffle=False, sampler= sampler)
            self.test_loader = GeometricDataLoader(test_dataset, batch_size=self.batchsize)
        else:
            self.train_loader = GeometricDataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)
            self.test_loader = GeometricDataLoader(test_dataset, batch_size=self.batchsize)

        if self.modeltype == "GIN":
            self.model = graphalgos.GIN(num_node_features=dataset.num_node_features, dim_h = self.hidden_channels).to(self.device)
        elif self.modeltype == "GCN":
            self.model = graphalgos.GCN(num_node_features=dataset.num_node_features, num_classes = dataset.num_classes, dim_h = self.hidden_channels).to(self.device)
        elif self.modeltype == "GAT":
            self.model = graphalgos.GAT(num_node_features=dataset.num_node_features, num_classes = dataset.num_classes, dim_h = self.hidden_channels, num_heads=self.heads).to(self.device)
        #select optimizer
        if self.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay = 10 ** (-5.0))
        elif self.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        y = dataset._data.y
        class_weights= compute_class_weight(class_weight='balanced', classes= np.unique(y.numpy()), y = y.numpy())
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        #weight = torch.tensor(self.posweight, dtype = torch.float32)
        if self.posweight: 
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to('cuda')
            #self.criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
        else:
            self.criterion = torch.nn.CrossEntropyLoss().to('cuda')
            #self.criterion = torch.nn.BCEWithLogitsLoss().to('cuda')
        # Train the model
        self.train()
        # Test the model
        self.test()
        torch.save(self.model, model_save_path)
        print(f"Model saved to {model_save_path}")