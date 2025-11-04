'''
In this script, I am going to sample using an LSTM to predict GPP using
meteorological variables at one given site. Eventually, I will also try an ESN
and expand this to multiple sites.
'''
import torch
import torch.nn as nn
import numpy as np
import pandas
import matplotlib.pyplot as plt

# Need to add data importation here

# Starting with site CA-Cbo, a deciduous broadleaf forest with 16 years of data
site = 'CA-Cbo'
data = df[df['Site']==site]

# Checking for continuity in the data (all good for CA-Cbo!)
data = data.sort_values("TIMESTAMP")
full_range = pd.date_range(data["TIMESTAMP"].min(), data["TIMESTAMP"].max(), freq="D")
missing = full_range.difference(data["TIMESTAMP"])
data.shape[0] == len(full_range)

# Set a random seed for reproducibility
np.random.seed(8675309)
torch.manual_seed(8675309)


def prepare_multivariate_data(X_df, y_series, lookback):
    """
    Prepares multivariate LSTM inputs.

    Parameters
    ----------
    X_df : pandas DataFrame or 2D numpy array
        Input features (each column = one feature)
    y_series : pandas Series or 1D numpy array
        Target variable (can be different from any X column)
    lookback : int
        Number of past time steps to include in each sample

    Returns
    -------
    X, y : numpy arrays ready for LSTM
        X.shape = (num_samples, lookback, num_features)
        y.shape = (num_samples,)
    """
    X, y = [], []
    for i in range(len(X_df) - lookback):
        X.append(X_df.iloc[i:i+lookback].values)
        y.append(y_series.iloc[i+lookback])      
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        '''
        Description: This initializes the LSTM according to our specifications.
        
        Parameters
        ----------
        input_dim : TYPE
            DESCRIPTION. Number of features in the input sequence at each time step
        hidden_dim : TYPE
            DESCRIPTION. Number of features in the hidden state.
        layer_dim : TYPE
            DESCRIPTION. Number of layers.
        output_dim : TYPE
            DESCRIPTION. Number of LSTM layer we want to stack.

        Returns
        -------
        None.

        '''
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h0=None, c0=None):
        '''
        Description: This performs a forward pass through the LSTM and fully connected layer.
        '''
        # Initialize hidden and cell states as 0 if they don't exist yet
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        
        # Get the hidden states for every timestep (out), the last hidden state 
        # from each layer (hn), and the last cell state from each layer (cn)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Select last prediction as the output
        out = self.fc(out[:, -1, :])
        
        return out, hn, cn

# Starting with X inputs as temperature, shortwave, vpd, prec, and swc
x_measures = ["TA_F","SW_IN_F","VPD_F","P_F","SWC_F_MDS_1"]
X_unstructured = data[x_measures] # confirm this is a pandas DataFrame
type(X_unstructured)

# we will predict for GPP
y_measure = "GPP_NT_VUT_REF"
y_unstructured = data[y_measure]
type(y_unstructured) # confirm this is a Series

# Preparing data structured for LSTM, with a lookback (or sequence length) of one month
X, y = prepare_multivariate_data(X_unstructured, y_unstructured, lookback=30)

# Transform into a tensor and split into training and testing data
trainX = torch.tensor(X, dtype=torch.float32)
trainX.shape # should be [batch_size, seq_length, input_dim]

trainY = torch.tensor(y[:,None], dtype=torch.float32)

# Initialize the model with 5 inputs, 50 hidden units, 2 layers, and 1 output
model = LSTMModel(input_dim=5, hidden_dim=50, layer_dim=2, output_dim=1)

# Set criterion to MSE loss function
criterion = nn.MSELoss()

# Set optimizer to Adams optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the LSTM model for 100 epochs
num_epochs = 100
h0, c0 = None, None
all_loss = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Reset states
    h0, c0 = None, None
    # Perform a forward pass with current initial states
    outputs, h0, c0 = model(trainX, h0, c0)
    loss = criterion(outputs, trainY)
    # Perform back propagation using the loss
    loss.backward()
    # Take a step towards the gradient
    optimizer.step()
    # Cut the states from the computational graph
    # h0, c0 = h0.detach(), c0.detach()
    # Add loss to the list
    all_loss.append(loss.detach())
    # Print the loss after every 10th epoch
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        

# Plotting loss over epochs
fig, ax = plt.subplots()
epochs = list(range(0,100))
plt.plot(epochs,all_loss)
plt.show()

# Plotting predicted vs observed
model.eval()
predicted, _, _ = model(trainX, h0, c0)

seq_length = 30
original = data['GPP_NT_VUT_REF'].iloc[seq_length:]
time_steps = np.arange(seq_length, len(data))

plt.figure(figsize=(12,6))
plt.scatter(time_steps, original, label="Original Data",s=.5)
plt.scatter(time_steps, predicted.detach().numpy(),
         label='Predicted Data', linestyle='--',s=.5)
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting observed vs predicted
x = np.linspace(0,50,10)
y = np.linspace(0,50,10)
plt.figure(figsize=(12,6))
plt.scatter(original, predicted.detach().numpy(),s=.5)
plt.plot(x,y,color='red')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Observed')
plt.ylabel('Predictions')
plt.ylim(0,max(original))
plt.xlim(0,max(original))
plt.legend()
plt.show()

# Accuracy metrics
R2 = 1 - (sum((original-predicted.detach().numpy().squeeze())**2) / sum((original - original.mean())**2))
print(R2)
