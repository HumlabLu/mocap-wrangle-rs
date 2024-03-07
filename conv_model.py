import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch.optim as optim
from torchshape import tensorshape
import argparse
import time, datetime, json

# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "--epochs",        "-e", help="Num epochs",          default=  1, type=int )
parser.add_argument( "--batchsize",     "-b", help="Batch size",          default= 10, type=int )
parser.add_argument( "--seqlen",        "-s", help="Sequence length",     default= 20, type=int )
parser.add_argument( "--hidden",        "-H", help="Hidden units",        default=104, type=int )
parser.add_argument( "--layers",        "-l", help="Number of layers",    default=  1, type=int )
parser.add_argument( "--features",      "-f", help="Number of features",  default=  2, type=int )
parser.add_argument( "--targets",       "-T", help="Number of targets",   default=  1, type=int )
parser.add_argument( "--targetfeature", "-F", help="The target feature",  default=  1, type=int )
parser.add_argument( "--classes",       "-c", help="Number of classes",   default=  0, type=int )
parser.add_argument( "--nogain",        "-g", help="Epochs without gain", default= 10, type=int )
parser.add_argument( "--restart",       "-r", help="Restart training",    action="store_true" )
parser.add_argument( "--scale",         "-S", help="Scale data",          action="store_true" )
parser.add_argument( "--best",          "-B", help="Load best_model.pht", action="store_true" )
parser.add_argument( "--alldata",       "-a", help="test on all data",    action="store_true" )
parser.add_argument( "--noplots",       "-p", help="Show no plots",       action="store_true" )
parser.add_argument( "--trainfile",     "-t", help="Training file",       default="foo.tsv" )
parser.add_argument( "--testfile",            help="Test file",           default=None )
parser.add_argument( "--model",         "-m", help="Model name",          default="ConvTabularModelP" )
parser.add_argument( "--id",            "-i", help="Extra ID string",     default=None )
parser.add_argument( "--info",          "-I", help="Print info only",     action="store_true" )
args = parser.parse_args()

# ============================================================================
# Logger and logging
# ============================================================================

def log(*args, sep=' ', end='\n', file='conv_03.log'):
    with open(file, 'a') as f:
        message = sep.join(map(str, args)) + end
        f.write(message)
        
print( "START", time.asctime() )
print( args )
with open("./conv_03.log", "a") as f:
    f.write(datetime.datetime.now().strftime('START %Y%m%dT%H:%M:%S\n'))
    json.dump(args.__dict__, f)
    f.write("\n")

# ============================================================================
# Data loaders.
# ============================================================================

'''
(VENVPYTORCH) pberck@ip21-178 mocap %
head eaf_targets.tsv 
Frame Timestamp RHandIn_dN RHandIn_vN RHandIn_aN LHandIn_dN LHandIn_vN LHandIn_aN LHand 
0	0.0	0.0	0.0	0.58	0.0	0.0	0.65	0
1	0.005	0.07	0.07	0.76	0.07	0.07	0.86	0
2	0.01	0.07	0.07	0.59	0.07	0.07	0.65	0

last two are targets, first two are meta-info.
'''

class TabSeparatedDataset(Dataset):
    def __init__(self, filepath, seqlen):
        # Load the data
        self.df = pd.read_csv(filepath, sep='\t')
        self.seqlen = seqlen
        
        num_columns = self.df.shape[1]
        # sensors is a misnomer...
        self.num_features = num_columns - 2 - args.targets # - frame, TS and targets
        print( f"Number of features/data fields: {self.num_features}" )

        # Cut the feature values, assume last is target.
        self.features = self.df.iloc[:, 2:-1]
        self.labels   = self.df.iloc[:,   -1]

        # One hot encode the values using pandas.
        self.max_class = int(self.labels.max())+1
        self.labels = pd.get_dummies(self.labels).values.reshape(-1, self.max_class)
        print("labels shape", self.labels.shape)
        print(self.labels[0])
        
        # Reshape features to match the input shape (1, 28, 11)
        # We take the target of the last row as target, so we miss the first seqlen data.
        self.num_samples = len(self.df) - seqlen + 1

        slices = []
        # Loop through the DataFrame, starting each new slice one row later than the previous
        for start_row in range(len(self.df) - seqlen + 1):
            slice_2d = self.features.iloc[start_row:start_row + seqlen].values
            #print("slice 2d", slice_2d, slice_2d.shape)
            image = slice_2d.reshape(1, seqlen, self.num_features)
            slices.append(image)
        # Convert the list of 2D arrays into a new DataFrame
        self.features = np.concatenate(slices, axis=0)
        self.features = self.features.reshape(-1, 1, seqlen, self.num_features)
        #self.features = self.features[self.num_samples:-1].reshape(-1, 1, seqlen, num_features)
        print("features shape", self.features.shape)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve features and label
        features = torch.tensor(self.features[idx], dtype=torch.float)
        # The first "image" has the last label as target?
        label = torch.tensor(self.labels[idx+self.seqlen-1], dtype=torch.float)
        return features, label

    def get_num_features(self):
        return self.num_features

    def get_num_classes(self):
        return self.max_class

# ============================================================================
# Models.
# ============================================================================

# See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# for output size calculations!
# and https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
class ConvTabularModelP(nn.Module):
    def __init__(self, channels, height, width, features, hidden, pool1_size=3, pool2_size=2):
        # channels could be sensors, height is number of frames, width is sensor values
        super(ConvTabularModelP, self).__init__()
        # Input image: 1 x 28 x 11 (C x H x W), grayscale images
        image_size = (args.batchsize, channels, height, width)
        # 32 is out channels
        #
        # seqlen must be at least pool1_size * pool2_size.
        if height < pool1_size * pool2_size:
            pool1_size = 1
            pool2_size = 1
            print("Setting pool sizes to 1.")
            log("Setting pool sizes to 1.")
        if width < pool1_size * pool2_size:
            pool1_size = 1
            pool2_size = 1
            print("Setting pool sizes to 1.")
            log("Setting pool sizes to 1.")
        #
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)  # Output: 32 x height x width
        out_size = tensorshape(self.conv1, image_size)
        print("out_size conv1", out_size)
        log("out_size conv1", out_size)
        #
        self.pool1_size = pool1_size
        self.pool1 = nn.MaxPool2d(self.pool1_size, self.pool1_size)
        out_size = tensorshape(self.pool1, out_size)
        print("out_size pool1", out_size)
        log("out_size pool1", out_size)
        #
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x7x2
        out_size = tensorshape(self.conv2, out_size)
        print("out_size conv2", out_size)
        log("out_size conv2", out_size)
        #
        self.pool2_size = pool2_size
        self.pool2 = nn.MaxPool2d(self.pool2_size, self.pool2_size)
        out_size = tensorshape(self.pool2, out_size) # we run the same pool again!
        print("out_size pool2", out_size)
        log("out_size pool2", out_size)
        #
        self.flatten = nn.Flatten()
        out_size = tensorshape(self.flatten, out_size)
        print("out_size flatten", out_size)
        log("out_size flatten", out_size)
        #
        self.fc1 = nn.Linear(out_size[-1], hidden) 
        out_size = tensorshape(self.fc1, out_size)
        print("out_size fc1", out_size)
        log("out_size fc1", out_size)
        self.fc2 = nn.Linear(hidden, features)  # Second dense layer, outputting "features" classes
        out_size = tensorshape(self.fc2, out_size)
        print("out_size fc2", out_size)
        log("out_size fc2", out_size)

    def forward(self, x):
        x = self.conv1(x)
        ##print(x.shape)
        x = F.relu(x)
        ##print(x.shape)
        x = self.pool1(x)
        ##print(x.shape)
        ##print()
        #x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 -> ReLU -> Pool
        #
        x = self.conv2(x)
        ##print(x.shape)
        x = F.relu(x)
        ##print(x.shape)
        x = self.pool2(x) ## again?
        ##print(x.shape)
        #x = self.pool(F.relu(self.conv2(x)))  # Apply conv2 -> ReLU -> Pool
        #
        #x = x.view(-1, 64 * 7 * 2)  # Flatten the output for dense layers
        x = self.flatten(x)
        #
        x = F.relu(self.fc1(x))  # Apply fc1 -> ReLU
        x = self.fc2(x)  # Output layer
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        probabilities = F.softmax(logits, dim=1)
        return probabilities

# ============================================================================
# Code.
# ============================================================================


training_data = TabSeparatedDataset(args.trainfile, args.seqlen)

# Parameters for the batch of random values
batch_size = args.batchsize  # Number of images in the batch
channels = 1     # Number of channels per image (1 for grayscale)
height = args.seqlen      # Height of the images (number of frames)
width = training_data.get_num_features()      # Width of the images (sensor values)
print(batch_size, channels, height, width)

# Generate a batch of random input values
random_batch = torch.rand(batch_size, channels, height, width)
print("random_batch shape", random_batch.shape)
#print(random_batch)

# Instantiate the model
model = ConvTabularModelP(channels, height, width, training_data.get_num_classes(), args.hidden)
print(model)
log(model)

# Assuming random_batch is the batch of random values generated as shown previously
# Pass the batch through the model
output = model(random_batch)
print("Output values for the batch:")
print(output)

# To interpret these as probabilities (for classification tasks), you can apply softmax
probabilities = torch.softmax(output, dim=1)
print("\nProbabilities of the classes for the first item in the batch:")
print(probabilities[0])

#print(torch.softmax(probabilities, dim=1))

fv, lbl = training_data[0]
fv = fv.reshape(1, *fv.shape)
print("fv.shape", fv.shape)
print("fv", fv)
print("lbl.shape", lbl.shape)
print("lbl", lbl)

foo = model.predict(fv)
print(foo)

train_loader = DataLoader(training_data, batch_size=args.batchsize, shuffle=True)
X, y = next(iter(train_loader))
print(X.shape, y.shape)
print("len data loader/total batches", len(train_loader))


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1

# Device configuration - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for i, batch in enumerate(train_loader):
        # Get data to cuda if possible
        features = batch[0].to(device)
        labels = batch[1].to(device)

        # Reshape features to match the model's expected input dimensions
        #features = features.unsqueeze(1)  # Add channel dimension

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()  # Clear gradients from the previous iteration
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()

        if (i+1) % 10 == 0:  # Print average loss every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0



print( "END", time.asctime() )
print()
with open("./conv_03.log", "a") as f:
    f.write(datetime.datetime.now().strftime('END %Y%m%dT%H:%M:%S\n\n'))
