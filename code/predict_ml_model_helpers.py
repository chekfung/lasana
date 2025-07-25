# Common Packages for Data Analysis
import numpy as np
import pandas as pd
import os
import time
import joblib

# Regressors
from scipy.interpolate import NearestNDInterpolator
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neural_network import MLPRegressor

# Classifiers
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier
from scipy.interpolate import NearestNDInterpolator
from sklearn.linear_model import LogisticRegression

# Metrics and Statistics
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_percentage_error, mean_absolute_error    # regressor
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score    # classifier
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

# Pytorch additions :)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR


RANDOM_STATE = 42

def runwise_train_test_split(df, run_column='Run_Number', test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
    # Get unique run numbers
    unique_runs = df[run_column].unique()
    
    # Split the unique runs into train, test and validation runs
    train_val_runs, test_runs = train_test_split(unique_runs, test_size=test_size, random_state=RANDOM_STATE)
    train_runs, val_runs = train_test_split(train_val_runs, test_size=val_size, random_state=RANDOM_STATE)

    # Create train, test, and validation runs
    train_df = df[df[run_column].isin(train_runs)]
    test_df = df[df[run_column].isin(test_runs)]
    val_df = df[df[run_column].isin(val_runs)]

    return train_df, test_df, val_df

class TorchStandardScaler(nn.Module):
    def __init__(self):
        super(TorchStandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def fit(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self.std[self.std == 0] = 1.0  # Handle zero variance like sklearn

    def transform(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return (x - self.mean) / (self.std)

    def forward(self, x):
        return self.transform(x)  # Apply transform as part of the forward pass


def produce_or_load_common_standard_scalar(df, list_of_columns, ml_model_filepath, run_column="Run_Number", test_size=0.15, val_size=0.15, random_state=RANDOM_STATE, output_pytorch=False):
    # First check if there is already a standard scalar :O
    std_scalar_name = 'ml_standard_scalar_random_seed_' + str(random_state)
    joblib_scaler = os.path.join(ml_model_filepath, std_scalar_name + ".joblib")
    torch_scaler = os.path.join(ml_model_filepath, std_scalar_name + ".pth")


    bool_check = (output_pytorch and os.path.exists(torch_scaler)) | (not output_pytorch and os.path.exists(joblib_scaler))

    if bool_check:
        if output_pytorch:
            std_scaler = TorchStandardScaler()
            std_scaler.load(torch_scaler)
        else:
            std_scaler = joblib.load(joblib_scaler) 
        
    else:
        print("Making Std Scaler")
        # First split full DF after converting to ns, pJ and then train the standard scalar on the trainset.
        train_df, test_df, val_df = runwise_train_test_split(df, run_column, test_size, val_size, random_state)
        train_df = train_df[list_of_columns]
        train_df = train_df.to_numpy()

        # Grab train_df and then train a standard scalar on it :)
        joblib_std_scaler = StandardScaler()
        joblib_std_scaler.fit(train_df)

        # Save the standard scaler
        joblib.dump(joblib_std_scaler, joblib_scaler)

        # Create Pytorch one as well :)
        torch_scale = TorchStandardScaler()
        torch_scale.fit(train_df)
        torch.save(torch_scale, torch_scaler)

        if output_pytorch:
            std_scaler = torch_scale
        else:
            std_scaler = joblib_std_scaler
    
    # Return scaler
    return std_scaler

def remove_outliers_zscore(column, z_score):
    z_scores = (column - column.mean()) / column.std()
    return column[abs(z_scores) < z_score]  # Keeping values within 3 standard deviations

def calculate_metrics(y_true, y_pred, decimal_places=7):
    # Calculate MSE
    mse = mean_squared_error(y_true, y_pred)

    # Calculate MAE
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_true,y_pred) * 100    # Native sklearn not percentage :O

    # Calculate R-Squared
    r2 = r2_score(y_true, y_pred)
    
    

    # Calculate Total Predicted versus Total Real (This is for energy calculations which are additive.)
    average_percent_error = float(np.abs(np.sum(np.array(y_true)) - np.sum(np.array(y_pred))) / np.sum(np.array(y_true)) * 100)

    # Calculate Total Predicted versus Total Real (This is for energy calculations which are additive.)
    total_pred = np.sum(np.array(y_pred))
    total_real = np.sum(np.array(y_true))

    # Format the metrics with the specified number of decimal places
    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [mse, mae, mape, r2, average_percent_error, total_pred, total_real]]

    return formatted_metrics

def write_prettytable(file_name, table_obj, append=False):
    mode = 'a' if append and os.path.exists(file_name) else 'w'
    with open(file_name, mode, newline='') as f_output:
        f_output.write(table_obj.get_csv_string())

# ---------- BEGIN Nearest Neighbor Interpolation Methods ----------
def closest_inputs(train_set, input_vector):
    distances = np.linalg.norm(train_set - input_vector, axis=1)
    closest_indices = np.argsort(distances)[:2]
    return closest_indices

def interpolate(X_train, X_test, X_val, y_train, y_test, y_val):
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    start_time = time.time()
    interpolate = NearestNDInterpolator(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    interp = interpolate(X_test)

    # Calculate total inference time
    end_time = time.time()
    inference_time = end_time - start_time

    return interp, train_time, inference_time

# ---------- END Nearest Neighbor Interpolation Methods ----------

# ---------- BEGIN other ML Regression Methods Methods ----------

def train_linear_regression(X_train, X_test, X_val, y_train, y_test, y_val, std_scaler):
    # Initialize the LinearRegression model
    model = LinearRegression()

    # Train the model
    start_time = time.time()
    X_train_scaled = std_scaler.transform(X_train.to_numpy())
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    # Make predictions
    start_time = time.time()
    X_test_scaled = std_scaler.transform(X_test.to_numpy())
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time

    return y_pred, train_time, inference_time

def run_catboost_regression(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, save_model=False, model_name="", cpp_model=False, early_stopping=True):
    model = CatBoostRegressor(**hyperparams)

    start_time = time.time()
    if early_stopping:
        model.fit(X_train, y_train, eval_set=(X_val,y_val), early_stopping_rounds=50)
    else:
        model.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time

    if save_model:
        if cpp_model:
            model.save_model(model_name+'.cpp', format='cpp')
        else:
            model.save_model(model_name+'.cbm')

    return y_pred, train_time, inference_time

def train_mlp_regression(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, save_model=False, model_name=""):
    # Since Sklearn's MLP is weird, combine X_train and X_val back together
    # Combine train and validation sets into one dataframe
    combined_X_train = pd.concat([X_train, X_val])
    combined_y_val = np.concatenate([y_train, y_val])

    # Initialize the MLPRegressor with hyperparameters
    mlp = MLPRegressor(**hyperparams)

    # Train the model
    start_time = time.time()
    X_train_scaled = std_scaler.transform(combined_X_train.to_numpy())
    mlp.fit(X_train_scaled, combined_y_val)
    end_time = time.time()
    train_time = end_time - start_time
    
    # Make predictions
    start_time = time.time()
    X_test_scaled = std_scaler.transform(X_test.to_numpy())
    y_pred = mlp.predict(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time

    # Save model using joblib
    if save_model:
        joblib.dump(mlp, model_name+'.joblib')

    
    return y_pred, train_time, inference_time

# ---------- END other ML Regression Methods Methods ----------

# ---------- BEGIN ML Classification Methods Methods ----------

def calculate_binary_classification_metrics(y_true, y_pred, decimal_places=7):
    # Calculate Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate ROC AUC
    auc = roc_auc_score(y_true, y_pred)

    # Calculate Precision
    prec = precision_score(y_true, y_pred)

    # Recall
    recall = recall_score(y_true, y_pred)

    # F1
    f1 = 2 * (prec * recall) / (prec + recall)

    # MSE
    mse = mean_squared_error(y_true *1.5, y_pred*1.5) # Vdd or 0 for output of the spiking neuron
    #mse = mean_squared_error(y_true, y_pred)


    # Format the metrics with the specified number of decimal places
    format_string = "{:." + str(decimal_places) + "f}"
    formatted_metrics = [format_string.format(metric) for metric in [acc, auc, prec, recall, f1, mse]]

    return formatted_metrics

def calculate_logistic_regression(X_train, X_test, X_val, y_train, y_test, y_val, std_scaler):
    model = LogisticRegression()

    # Train model
    start_time = time.time()
    X_train_scaled = std_scaler.transform(X_train.to_numpy())
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    # Inference Time
    start_time = time.time()
    X_test_scaled = std_scaler.transform(X_test.to_numpy())
    y_pred = model.predict(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time

    return y_pred, train_time, inference_time

def train_svm(X_train, X_test, y_train, y_test, std_scaler, model_type='linear'):
    # Initialize the LinearRegression model
    svm = SVC(kernel=model_type)

    # Train the model
    start_time = time.time()
    X_train_scaled = std_scaler.transform(X_train.to_numpy())
    svm.fit(X_train_scaled, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    # Make predictions
    start_time = time.time()
    X_test_scaled = std_scaler.transform(X_test.to_numpy())
    y_pred = svm.predict(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time

    #print("Test MSE:", mse)
    return y_pred, train_time, inference_time

def run_catboost_classify(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, save_model=False, model_name="", cpp_model=False):
    model = CatBoostClassifier(**hyperparams)

    start_time = time.time()
    model.fit(X_train, y_train, eval_set=(X_val,y_val), early_stopping_rounds=50)
    end_time = time.time()
    train_time = end_time - start_time

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time

    if save_model:
        if cpp_model:
            model.save_model(model_name+'.cpp', format='cpp')

        else:
            model.save_model(model_name+'.cbm')

    return y_pred, train_time, inference_time

def train_mlp_classify(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, save_model=False, model_name=""):
    # Since Sklearn's MLP is weird, combine X_train and X_val back together
    # Combine train and validation sets into one dataframe
    combined_X_train = pd.concat([X_train, X_val])
    combined_y_val = np.concatenate([y_train, y_val])
    
    # Initialize the MLPRegressor with hyperparameters
    mlp = MLPClassifier(**hyperparams)

    # Train the model
    start_time = time.time()
    X_train_scaled = std_scaler.transform(combined_X_train.to_numpy())
    mlp.fit(X_train_scaled, combined_y_val)
    end_time = time.time()
    train_time = end_time - start_time
    
    # Make predictions
    start_time = time.time()
    X_test_scaled = std_scaler.transform(X_test.to_numpy())
    y_pred = mlp.predict(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time

    # Save model using joblib
    if save_model:
        joblib.dump(mlp, model_name+'.joblib')
    
    return y_pred, train_time, inference_time

# ---------- END ML Classification Methods Methods ----------

class MLPRegressorPyTorch(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_fn, output_dim=1):
        super(MLPRegressorPyTorch, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Select activation function based on the input hyperparameter
        self.activation = activation_fn
        
        # Create the hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(self.activation)  # Use the activation function
            prev_dim = hidden_size
        
        # Create the output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_pytorch_mlp(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, device='cpu', save_model=False, model_name=""):
    # Convert the data into PyTorch tensors
    X_train_scaled = torch.tensor(std_scaler.transform(X_train.to_numpy()), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.ravel(y_train), dtype=torch.float32).view(-1, 1).to(device)  # Ensuring the shape is correct
    X_test_scaled = torch.tensor(std_scaler.transform(X_test.to_numpy()), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(np.ravel(y_test), dtype=torch.float32).view(-1, 1).to(device)

    X_val_scaled = torch.tensor(std_scaler.transform(X_val.to_numpy()), dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(np.ravel(y_val), dtype=torch.float32).view(-1, 1).to(device)
    
    # Define hyperparameters and the model
    input_dim = X_train_scaled.shape[1]
    
    # Get the activation function from the hyperparameters (default to ReLU if not specified)
    activation_fn = hyperparams.get('activation', nn.ReLU())  # Default to ReLU
    model = MLPRegressorPyTorch(input_dim, hidden_layers=hyperparams['hidden_layer_sizes'], activation_fn=activation_fn).to(device)
    
    # Define loss function and optimizer
    criterion = hyperparams.get('loss_fn', nn.MSELoss())  # Default to MSELoss if not specified
    optimizer = hyperparams.get('optimizer', optim.Adam(model.parameters(), lr=hyperparams['learning_rate_init'], weight_decay=hyperparams['alpha']))
    
    # Convert to DataLoader for batching
    batch_size = hyperparams.get('batch_size', 32)
    train_data = TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Train the model
    start_time = time.time()
    model.train()  # Set the model to training mode
    num_epochs = hyperparams.get('num_epochs', 100)  # Number of epochs
    prev_val_loss = -1 * np.inf

    epoch_loss = []
    val_loss = []


    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        print(f"Training Epoch : {epoch} / {num_epochs}")
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss_per_epoch+=loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss.append(loss_per_epoch)

        model.eval()
        # Run Validation set and check loss :)
        y_val_pred = model(X_val_scaled)
        val_loss_epoch = criterion(y_val_pred, y_val_tensor)
        val_loss_diff = abs(prev_val_loss - val_loss_epoch)
        print(f"Epoch: {epoch} / {num_epochs}, Validation Loss: {val_loss_epoch:.5f}, Validation Loss Difference: {val_loss_diff}")
        conv_tol = hyperparams.get('tol', 1e-4)
        val_loss.append(val_loss_epoch.item())

        if val_loss_diff < conv_tol:
            print(f"Earlying Stopping because convergence tolerance reached of: {conv_tol}")
            break
            
        prev_val_loss = val_loss_epoch

    end_time = time.time()
    train_time = end_time - start_time

    # Create the figure and the axes (2 subplots, 1 row and 2 columns)
    plt.figure(1000)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Train Loss on the first subplot (ax[0])
    ax[0].plot(epoch_loss, label='train loss', color='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[0].legend()

    # Plot Validation Loss on the second subplot (ax[1])
    ax[1].plot(val_loss, label='val loss', color='red')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Validation Loss")
    ax[1].legend()

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    
    # Make predictions
    start_time = time.time()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred = model(X_test_scaled)
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Save the model
    if save_model:
        torch.save(model, model_name + ".pth")
    
    return y_pred.cpu().numpy(), train_time, inference_time
# ----
class MLPClassifierPyTorch(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, activation_fn, output_activation=None):
        super(MLPClassifierPyTorch, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Select activation function based on the input hyperparameter
        self.activation = activation_fn
        
        # Create the hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(self.activation)  # Use the activation function
            prev_dim = hidden_size
        
        # Create the output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Final activation for output layer (Softmax for multi-class, Sigmoid for binary)
        if output_activation is not None:
            layers.append(output_activation)
        
        # Combine all layers
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def train_pytorch_mlp_classifier(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, device='cpu', save_model=False, model_name=""):
    # Convert the data into PyTorch tensors
    X_train_scaled = torch.tensor(std_scaler.transform(X_train.to_numpy()), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(np.ravel(y_train), dtype=torch.float32).view(-1, 1).to(device)  # Ensuring the shape is correct
    X_test_scaled = torch.tensor(std_scaler.transform(X_test.to_numpy()), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(np.ravel(y_test), dtype=torch.float32).view(-1, 1).to(device)

    X_val_scaled = torch.tensor(std_scaler.transform(X_val.to_numpy()), dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(np.ravel(y_val), dtype=torch.float32).view(-1, 1).to(device)
    
    # Define hyperparameters and the model
    input_dim = X_train_scaled.shape[1]
    
    # Get the activation function from the hyperparameters (default to ReLU if not specified)
    activation_fn = hyperparams.get('activation', nn.ReLU())  # Default to ReLU
    output_dim = 1  # Number of Output Predictions
    output_activation = None#nn.Sigmoid()  # Softmax is handled internally by CrossEntropyLoss
    
    model = MLPClassifierPyTorch(input_dim, hidden_layers=hyperparams['hidden_layer_sizes'], 
                                 output_dim=output_dim, activation_fn=activation_fn, 
                                 output_activation=output_activation).to(device)
    
    # Define loss function and optimizer
    criterion = hyperparams.get('loss_fn', nn.BCEWithLogitsLoss())  # Default to CrossEntropyLoss

    optimizer = hyperparams.get('optimizer', optim.Adam(model.parameters(), lr=hyperparams['learning_rate_init'], weight_decay=hyperparams['alpha']))
    
    # Convert to DataLoader for batching
    batch_size = hyperparams.get('batch_size', 32)
    train_data = TensorDataset(X_train_scaled, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Train the model
    start_time = time.time()
    model.train()  # Set the model to training mode
    num_epochs = hyperparams.get('num_epochs', 100)  # Number of epochs

    prev_val_loss = -1 * np.inf
    epoch_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        print(f"Training Epoch : {epoch} / {num_epochs}")
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch) 
            logits = logits.clamp(-10, 10)  # Clamp

            loss = criterion(logits, y_batch)
            loss_per_epoch+=loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss.append(loss_per_epoch)

        model.eval()
        # Run Validation set and check loss :)
        y_val_pred = model(X_val_scaled)
        y_val_pred = y_val_pred.clamp(-10, 10)  # Clamp
        val_loss_epoch = criterion(y_val_pred, y_val_tensor)
        val_loss_diff = abs(prev_val_loss - val_loss_epoch)
        print(f"Epoch: {epoch} / {num_epochs}, Validation Loss: {val_loss_epoch:.5f}, Validation Loss Difference: {val_loss_diff}")
        conv_tol = hyperparams.get('tol', 1e-4)
        val_loss.append(val_loss_epoch.item())

        if val_loss_diff < conv_tol:
            print(f"Earlying Stopping because convergence tolerance reached of: {conv_tol}")
            break
            
        prev_val_loss = val_loss_epoch

    end_time = time.time()
    train_time = end_time - start_time

    # Create the figure and the axes (2 subplots, 1 row and 2 columns)
    plt.figure(1000)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Train Loss on the first subplot (ax[0])
    ax[0].plot(epoch_loss, label='train loss', color='blue')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[0].legend()

    # Plot Validation Loss on the second subplot (ax[1])
    ax[1].plot(val_loss, label='val loss', color='red')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Validation Loss")
    ax[1].legend()

    # Adjust the layout to avoid overlap
    plt.tight_layout()
    # Make predictions
    start_time = time.time()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_pred = model(X_test_scaled)
        y_pred = y_pred.clamp(-10, 10)  # Clamp
        
        predicted_classes = (y_pred > 0).int()  # Get class predictions

    end_time = time.time()
    inference_time = end_time - start_time
    
    # Save the model
    if save_model:
        torch.save(model, model_name + ".pth")
    
    return predicted_classes.cpu().numpy(), train_time, inference_time




# Try and predict multitask MLP for pytorch to make computation graph simpler :)
class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_fn):
        super(MultiTaskMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        self.activation = activation_fn

        # Shared feature extractor
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(self.activation)
            prev_dim = hidden_size
        
        self.shared = nn.Sequential(*layers)

        # Output heads
        self.spike_head = nn.Linear(prev_dim, 1)  # Binary classification (Sigmoid)
        self.membrane_head = nn.Linear(prev_dim, 1)  # Regression (Linear)
        self.energy_head = nn.Linear(prev_dim, 1)  # Regression (Linear)
        self.latency_head = nn.Linear(prev_dim, 1)  # Regression (Linear)

    def forward(self, x):
        shared_features = self.shared(x)

        spike_prob = torch.sigmoid(self.spike_head(shared_features))
        membrane_potential = self.membrane_head(shared_features)
        energy = self.energy_head(shared_features)
        latency = self.latency_head(shared_features)

        return torch.cat((spike_prob, membrane_potential, energy, latency), dim=1)

def train_multi_task_model(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, device='cpu', save_model=False, model_name=""):
    # Convert the data into PyTorch tensors
    X_train_scaled = torch.tensor(std_scaler.transform(X_train.to_numpy()), dtype=torch.float32).to(device)
    X_test_scaled = torch.tensor(std_scaler.transform(X_test.to_numpy()), dtype=torch.float32).to(device)
    X_val_scaled = torch.tensor(std_scaler.transform(X_val.to_numpy()), dtype=torch.float32).to(device)

    # Convert multi-task targets to tensors
    # Convert the entire DataFrame to a NumPy array first
    y_train_np = y_train.to_numpy().astype(np.float32)
    y_test_np = y_test.to_numpy().astype(np.float32)
    y_val_np = y_val.to_numpy().astype(np.float32)

    # Convert NumPy arrays to PyTorch tensors and move to device
    y_train_tensors = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    y_test_tensors = torch.tensor(y_test_np, dtype=torch.float32).to(device)
    y_val_tensors = torch.tensor(y_val_np, dtype=torch.float32).to(device)
    print(y_train_tensors.shape)
    print(X_train_scaled.shape)
        
    # Define hyperparameters and the model
    input_dim = X_train_scaled.shape[1]
    print(input_dim)
    activation_fn = hyperparams.get('activation', nn.ReLU())
    hidden_layers = hyperparams['hidden_layer_sizes']

    model = MultiTaskMLP(input_dim, hidden_layers, activation_fn).to(device)

    # Define loss functions
    loss_fn_spike = nn.BCELoss()  # Binary cross-entropy loss for spike probability
    loss_fn_regression = nn.MSELoss()  # Mean squared error for all regression outputs

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate_init'], weight_decay=hyperparams['alpha'])

    # Convert to DataLoader
    batch_size = hyperparams.get('batch_size', 32)
    train_data = TensorDataset(X_train_scaled, y_train_tensors)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Training loop
    num_epochs = hyperparams.get('num_epochs', 100)
    prev_val_loss = float('inf')
    epoch_loss = []
    val_loss = []

    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        print(f"Training Epoch: {epoch} / {num_epochs}")

        for batch in train_loader:
            X_batch, y_batch = batch
            optimizer.zero_grad()

            y_pred = model(X_batch)
    
            loss_spike = loss_fn_spike(y_pred[:,0], y_batch[:,0])  # Spike probability loss
            loss_memb = loss_fn_regression(y_pred[:,1], y_batch[:,1])  # Membrane potential loss
            loss_energy = loss_fn_regression(y_pred[:,2], y_batch[:,2])  # Dynamic energy loss
            loss_latency = loss_fn_regression(y_pred[:,3], y_batch[:,3])  # Latency loss

            # Combined loss (weighted sum, adjust weights if necessary)
            total_loss = loss_spike + loss_memb + loss_energy + loss_latency
            total_loss.backward()
            optimizer.step()

            loss_per_epoch += total_loss.item()

        epoch_loss.append(loss_per_epoch)

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_scaled)

            # Compute individual losses
            spike_loss = loss_fn_spike(y_val_pred[:, 0], y_val_tensors[:, 0])
            membrane_loss = loss_fn_regression(y_val_pred[:, 1], y_val_tensors[:, 1])
            dynamic_energy_loss = loss_fn_regression(y_val_pred[:, 2], y_val_tensors[:, 2])
            latency_loss = loss_fn_regression(y_val_pred[:, 3], y_val_tensors[:, 3])

            # Sum total loss
            val_loss_epoch = spike_loss + membrane_loss + dynamic_energy_loss + latency_loss

        val_loss.append(val_loss_epoch.item())

        # Print all losses on one line
        print(f"Epoch: {epoch} | Total Validation Loss: {val_loss_epoch:.5f} | Spike Loss: {spike_loss:.5f} | Membrane Potential Loss: {membrane_loss:.5f} | Dynamic Energy Loss: {dynamic_energy_loss:.5f} | Latency Loss: {latency_loss:.5f}")


        if abs(prev_val_loss - val_loss_epoch) < hyperparams.get('tol', 1e-4):
            print(f"Early stopping at epoch {epoch}")
            break

        prev_val_loss = val_loss_epoch

    end_time = time.time()
    train_time = end_time - start_time

    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Model inference on test set
    inference_start = time.time()
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_scaled)
        predicted_classes = (y_test_pred[:,0] > 0.5).int()  # Spike probability thresholding

        # Extract other continuous outputs
        membrane_potentials = y_test_pred[:,1].cpu().numpy()
        energy = y_test_pred[:,2].cpu().numpy()
        latency = y_test_pred[:,3].cpu().numpy()

        # Stack all predictions into a single NumPy array (shape: [num_samples, 4])
        combined_predictions = np.column_stack((predicted_classes, membrane_potentials, energy, latency))
        print(combined_predictions.shape)

    inference_time = time.time() - inference_start

    if save_model:
        torch.save(model, model_name + ".pth")

    return combined_predictions, train_time, inference_time

class BehavioralMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_fn):
        super(BehavioralMLP, self).__init__()

        layers = []
        prev_dim = input_dim
        self.activation = activation_fn

        # Shared feature extractor
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(self.activation)
            prev_dim = hidden_size

        self.shared = nn.Sequential(*layers)

        # Output heads (only spike and membrane)
        self.spike_head = nn.Linear(prev_dim, 1)  # Binary classification (Sigmoid)
        self.membrane_head = nn.Linear(prev_dim, 1)  # Regression (Linear)

    def forward(self, x):
        shared_features = self.shared(x)

        spike_prob = torch.sigmoid(self.spike_head(shared_features))
        membrane_potential = self.membrane_head(shared_features)

        return torch.cat((spike_prob, membrane_potential), dim=1)

def train_behavioral_model(X_train, X_test, X_val, y_train, y_test, y_val, hyperparams, std_scaler, device='cpu', save_model=False, model_name=""):
    # Convert the data into PyTorch tensors
    X_train_scaled = torch.tensor(std_scaler.transform(X_train.to_numpy()), dtype=torch.float32).to(device)
    X_test_scaled = torch.tensor(std_scaler.transform(X_test.to_numpy()), dtype=torch.float32).to(device)
    X_val_scaled = torch.tensor(std_scaler.transform(X_val.to_numpy()), dtype=torch.float32).to(device)

    # Convert multi-task targets to tensors
    y_train_np = y_train.to_numpy().astype(np.float32)
    y_test_np = y_test.to_numpy().astype(np.float32)
    y_val_np = y_val.to_numpy().astype(np.float32)

    # Convert NumPy arrays to PyTorch tensors and move to device
    y_train_tensors = torch.tensor(y_train_np, dtype=torch.float32).to(device)
    y_test_tensors = torch.tensor(y_test_np, dtype=torch.float32).to(device)
    y_val_tensors = torch.tensor(y_val_np, dtype=torch.float32).to(device)
        
    # Define hyperparameters and the model
    input_dim = X_train_scaled.shape[1]
    activation_fn = hyperparams.get('activation', nn.ReLU())
    hidden_layers = hyperparams['hidden_layer_sizes']

    model = BehavioralMLP(input_dim, hidden_layers, activation_fn).to(device)

    # Define loss functions
    loss_fn_spike = nn.BCELoss()  # Binary cross-entropy loss for spike probability
    loss_fn_regression = nn.MSELoss()  # Mean squared error for membrane potential

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate_init'], weight_decay=hyperparams['alpha'])

    # Convert to DataLoader
    batch_size = hyperparams.get('batch_size', 32)
    train_data = TensorDataset(X_train_scaled, y_train_tensors)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Training loop
    num_epochs = hyperparams.get('num_epochs', 100)
    prev_val_loss = float('inf')
    epoch_loss = []
    val_loss = []

    model.train()
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        loss_per_epoch = 0
        print(f"Training Epoch: {epoch} / {num_epochs}")

        for batch in train_loader:
            X_batch, y_batch = batch
            optimizer.zero_grad()

            y_pred = model(X_batch)

            # Spike and membrane losses
            loss_spike = loss_fn_spike(y_pred[:,0], y_batch[:,0])  # Spike probability loss
            loss_memb = loss_fn_regression(y_pred[:,1], y_batch[:,1])  # Membrane potential loss

            # Combined loss
            total_loss = loss_spike + loss_memb
            total_loss.backward()
            optimizer.step()

            loss_per_epoch += total_loss.item()

        epoch_loss.append(loss_per_epoch)

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_scaled)

            # Compute individual losses
            spike_loss = loss_fn_spike(y_val_pred[:, 0], y_val_tensors[:, 0])
            membrane_loss = loss_fn_regression(y_val_pred[:, 1], y_val_tensors[:, 1])

            # Sum total loss
            val_loss_epoch = spike_loss + membrane_loss

        val_loss.append(val_loss_epoch.item())

        # Print all losses on one line
        print(f"Epoch: {epoch} | Total Validation Loss: {val_loss_epoch:.5f} | Spike Loss: {spike_loss:.5f} | Membrane Potential Loss: {membrane_loss:.5f}")

        if abs(prev_val_loss - val_loss_epoch) < hyperparams.get('tol', 1e-4):
            print(f"Early stopping at epoch {epoch}")
            break

        prev_val_loss = val_loss_epoch

    end_time = time.time()
    train_time = end_time - start_time

    # Plot loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(epoch_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

    # Model inference on test set
    inference_start = time.time()
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_scaled)
        predicted_classes = (y_test_pred[:,0] > 0.5).int()  # Spike probability thresholding

        # Extract other continuous outputs
        membrane_potentials = y_test_pred[:,1].cpu().numpy()

        # Stack all predictions into a single NumPy array (shape: [num_samples, 2])
        combined_predictions = np.column_stack((predicted_classes, membrane_potentials))
        print(combined_predictions.shape)

    inference_time = time.time() - inference_start

    if save_model:
        torch.save(model, model_name + ".pth")

    return combined_predictions, train_time, inference_time
