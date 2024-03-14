import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import torch.optim as optim
from torchshape import tensorshape
import argparse
import time, datetime, json
import matplotlib.pyplot as plt
import matplotlib as mp
from tqdm import tqdm
import seaborn as sn
from torchvision import utils

# ============================================================================
# Arguments
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument( "--epochs",        "-e", help="Num epochs",          default=  1, type=int )
parser.add_argument( "--batchsize",     "-b", help="Batch size",          default= 10, type=int )
parser.add_argument( "--seqlen",        "-s", help="Sequence length",     default= 20, type=int )
parser.add_argument( "--hidden",        "-H", help="Hidden units",        default=108, type=int )
parser.add_argument( "--pooling",       "-p", help="Pooling tuple",       default=[1,1,1,1], nargs='+', type=int)
#parser.add_argument( "--layers",        "-l", help="Number of layers",    default=  1, type=int )
parser.add_argument( "--targets",       "-T", help="Number of targets",   default=  1, type=int )
#parser.add_argument( "--noplots",       "-p", help="Show no plots",       action="store_true" )
parser.add_argument( "--trainfile",     "-t", help="Training file",       default="foo.tsv" )
parser.add_argument( "--testfile",            help="Test file",           default=None )
parser.add_argument( "--unlabelled",    "-u", help="Unlabelled file",     default=None )
parser.add_argument( "--restart",       "-r", help="Restart training",    action="store_true" )
parser.add_argument( "--model",         "-m", help="Model name",          default="ConvTabularModelP" )
parser.add_argument( "--id",            "-i", help="Extra ID string",     default=None )
parser.add_argument( "--info",          "-I", help="Print info only",     action="store_true" )
args = parser.parse_args()

# ============================================================================
# Logger and logging
# ============================================================================

def log(*args, sep=' ', end='\n', file='conv_04.log'):
    with open(file, 'a') as f:
        message = sep.join(map(str, args)) + end
        f.write(message)
        
print( "START", time.asctime() )
print( args )
with open("./conv_04.log", "a") as f:
    f.write(datetime.datetime.now().strftime('START %Y%m%dT%H:%M:%S\n'))
    json.dump(args.__dict__, f)
    f.write("\n")

def ms_to_time(ms):
    secs = int(ms / 1000)
    millis = ms - (secs*1000)
    mins = int(secs / 60)
    secs = secs - (mins*60)
    return f"{mins:02}:{secs:02}.{millis:03}"

def visualise_conv1(model):
    kernels = model.conv1.weight.detach().clone()
    print("kernels shape:", kernels.numpy().shape)
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = utils.make_grid(kernels, nrow = 8) #, normalize=True, value_range=(0,255))
    print("filter_img shape:", filter_img.shape)
    # change ordering since matplotlib requires images to 
    # be (H, W, C)
    fig, ax = plt.subplots()
    im = ax.imshow(filter_img.permute(1, 2, 0))
    ax.set_title("Filters in conv1.")
    return fig, im
    #img = save_image(kernels, 'encoder_conv1_filters.png', nrow = 12)
    #sys.exit(0)

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

Last is target, first two are meta-info.

python eaf_extract.py -d gestures_ML_05_data.tsv -e gestures_ML_05.eaf -F "LHand" -t LHand -N -D -V -A -Z -o quuz.tsv

Frame	Timestamp	LHandIn_az	LHandIn_in	LHandIn_dN	LHandIn_vN	LHandIn_aN	LHandOut_az	LHandOut_in	LHandOut_dN	LHandOut_vN	LHandOut_aN	LHand
0	0.0	0.0	0.0	0.0	0.0	0.65	0.0	0.0	0.0	0.0	0.62	0
1	0.005	-1.7	2.5	0.07	0.07	0.86	-2.1	2.4	0.07	0.07	0.8	0
2	0.01	-1.9	2.6	0.07	0.07	0.65	-2.0	2.3	0.06	0.06	0.59	0
'''

def load_data(filepath, seqlen, getlabels=True, enc=None):
    df = pd.read_csv(filepath, sep='\t')
    
    num_columns = df.shape[1]
    # sensors is a misnomer...
    num_features = num_columns - 2 - args.targets # subtract frame, TS and targets
    print( f"Number of features/data fields: {num_features}" )

    # Cut the feature values, assume last is target.
    features = df.iloc[:, 2:-1]
    print(features.info())
    print(features.min(axis=0))
    print(features.max(axis=0))
    
    # The labels are in the last column. What about unlabelled data?
    if getlabels:
        labels = df.iloc[:, -1].values.reshape(-1, 1) ## This reshape is wrong for 1 dim labels!
        
    # One hot encode the values using sklearn. Parameter so we
    # can re-use the encoder for another file.
    if getlabels:
        if not enc:
            enc = OneHotEncoder(handle_unknown='ignore')
        #enc.fit(labels)
        enc_df = pd.DataFrame(enc.fit_transform(labels).toarray())
        #print(enc_df.head())
        labels = enc_df.iloc[seqlen-1:] # remove first seqlen because we predict last row of image
        print("labels shape", labels.shape)
        #print(labels.head())
    else:
        print("No labels.")
    
    # Reshape features to match the input shape (1, 28, 11)
    # We take the target of the last row as target, so we miss the first seqlen data.
    num_samples = len(df) - seqlen + 1

    slices = []
    # Loop through the DataFrame, starting each new slice one row later than the previous.
    # (Should we take bigger steps?)
    for start_row in range(len(df) - seqlen + 1):
        slice_2d = features.iloc[start_row:start_row + seqlen].values
        #print("slice 2d", slice_2d, slice_2d.shape)
        image = slice_2d.reshape(1, seqlen, num_features)
        slices.append(image)
    # Convert the list of 2D arrays into a new DataFrame
    features = np.concatenate(slices, axis=0)
    print("concatenated features shape", features.shape)
    features = features.reshape(-1, 1, seqlen, num_features)
    #features = features[num_samples:-1].reshape(-1, 1, seqlen, num_features)
    print("features shape", features.shape)

    #print("shapes", train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    #return (train_data, test_data, train_labels, test_labels, enc)
    if getlabels:
        return (features, labels, enc)
    return features

class TabSeparatedDataset(Dataset):
    def __init__(self, features, labels):
        """
        Args:
            features (numpy array): Features from the dataset.
            labels (numpy array): Labels corresponding to each row of features.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        #print(idx, self.features.shape, self.labels.shape)
        #print(self.features[idx])
        #print(self.labels.values[idx])

        features = torch.tensor(self.features[idx], device=device, dtype=torch.float)
        label = torch.tensor(self.labels.values[idx], device=device, dtype=torch.float)
        #print("getitem", features.shape, label.shape)
        return features, label
    
    def get_num_features(self):
        print("get_num_features", self.features.shape[3])
        return self.features.shape[3]

    def get_size(self):
        return self.features.shape[0]

    def get_num_classes(self):
        return self.labels.shape[-1]

# This is for unlabelled test data, or new data we want to classify.
class UnlabelledDataset(Dataset):
    def __init__(self, features):
        """
        Args:
            features (numpy array): Features from the dataset.
        """
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        #print(idx, self.features.shape, self.labels.shape)
        #print(self.features[idx])
        #print(self.labels.values[idx])

        features = torch.tensor(self.features[idx], dtype=torch.float)
        return features
    
    def get_num_features(self):
        print("get_num_features", self.features.shape[3])
        return self.features.shape[3]

    def get_size(self):
        return self.features.shape[0]

    def get_num_classes(self):
        return self.labels.shape[-1]

# ============================================================================
# Convolutional model.
# ============================================================================

# See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
# for output size calculations!
# and https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
class ConvTabularModelP(nn.Module):
    def __init__(self, channels, height, width, features, hidden, pooling):
        # channels could be sensors, height is number of frames, width is sensor values
        super(ConvTabularModelP, self).__init__()
        image_size = (args.batchsize, channels, height, width) # Channels is one  ("greyscale").
        # 32 is out channels
        #
        try:
            pool1_size = (pooling[0], pooling[1])
            pool1_stride = (1,1)
            pool2_size = (pooling[2], pooling[3])
            pool2_stride = (1,1)
        except:
            log("Wrong pooling")
            print("Wrong pooling")
            sys.exit(0)
        #
        filters1 = 16
        filters2 = 32
        #self.conv1 = nn.Conv2d(channels, filters1, kernel_size=(20, 10), padding=(20,10))  # 32 x height x width
        self.conv1 = nn.Conv2d(channels, filters1, kernel_size=3, padding=1)  # Output: 32 x height x width
        out_size = tensorshape(self.conv1, image_size)
        print("out_size conv1", out_size)
        log("out_size conv1", out_size)
        #
        self.pool1_size = pool1_size
        ##self.pool1 = nn.MaxPool2d(self.pool1_size, self.pool1_size)
        self.pool1 = nn.MaxPool2d(pool1_size, pool1_stride)
        out_size = tensorshape(self.pool1, out_size)
        print("out_size pool1", out_size)
        log("out_size pool1", out_size)
        #
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=3, padding=1)  # Output: 64x7x2
        out_size = tensorshape(self.conv2, out_size)
        print("out_size conv2", out_size)
        log("out_size conv2", out_size)
        #
        self.pool2_size = pool2_size
        ##self.pool2 = nn.MaxPool2d(self.pool2_size, self.pool2_size)
        self.pool2 = nn.MaxPool2d(pool2_size, pool2_stride)
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
        x = x.to(device)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        #x = self.pool(F.relu(self.conv1(x)))  # Apply conv1 -> ReLU -> Pool
        #
        x = self.conv2(x)
        x = F.relu(x) # was F.leaky_relu(x)
        x = self.pool2(x) 
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
# Training and test data.
# ============================================================================

features, labels, oh_enc = load_data(args.trainfile, args.seqlen) # return encoder for later.
train_data, test_data, train_labels, test_labels = train_test_split(features,
                                                                    labels,
                                                                    test_size=0.3,
                                                                    shuffle=True,
                                                                    random_state=42)
print("Train/test shapes", train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
#print(oh_enc)
#print(oh_enc.inverse_transform([[0, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 1, 0]]))

training_data = TabSeparatedDataset(train_data, train_labels)
testing_data  = TabSeparatedDataset(test_data, test_labels)

# ============================================================================
# Model.
# ============================================================================

# Parameters for the batch of random values
batch_size = args.batchsize  # Number of images in the batch
channels   = 1     # Number of channels per image (1 for grayscale)
height     = args.seqlen      # Height of the images (number of "mocap frames")
width      = training_data.get_num_features()      # Width of the images (sensor values)
print("batch_size, channels, height, width", batch_size, channels, height, width)

# Pooling/stride?

model = ConvTabularModelP(channels, height, width, training_data.get_num_classes(), args.hidden, args.pooling)
print(model)
log(model)

# Assuming random_batch is the batch of random values generated as shown previously
# Pass the batch through the model
#output = model(random_batch)
#print("Output values for the batch:")
#print(output)

# To interpret these as probabilities (for classification tasks), you can apply softmax
#probabilities = torch.softmax(output, dim=1)
#print("\nProbabilities of the classes for the first item in the batch:")
#print(probabilities[0])

#print(torch.softmax(probabilities, dim=1))

train_loader = DataLoader(training_data, batch_size=args.batchsize, shuffle=True)
test_loader  = DataLoader(testing_data, batch_size=args.batchsize, shuffle=False) 

# ============================================================================
# Visualise.
# ============================================================================

if args.info:
    hor=4
    ver=3
    f, axs = plt.subplots(ver, hor)
    for h in range(ver):
        for v in range(hor): #, v in zip(range(ver), range(hor)): #  [(0, 0), (0, 1), (1, 0), (1, 1)]:
            X, y = next(iter(train_loader)) # We need to reset it...
            xx = axs[h, v].imshow(np.abs(X[0][0]), cmap = 'gray', aspect='auto') # Used for colourbar.
            res_lbl = oh_enc.inverse_transform(y[0].reshape(1, -1))
            axs[h, v].set_title(res_lbl)
    #bar = plt.colorbar(xx) # Will be for the last image.
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(xx, cax=cbar_ax)
    #bar.set_label('ColorBar 1')
    f.subplots_adjust(hspace=0.5)
    plt.show()
    png_filename = os.path.basename(args.trainfile)+"_s"+str(args.seqlen)+".imgdata.png"
    print( "Saving", png_filename )
    f.savefig(png_filename, dpi=288)
    sys.exit(0)

# ============================================================================
# Training.
# ============================================================================

try:
    pooling_str = "_p"+str(args.pooling[0])+str(args.pooling[1])+str(args.pooling[2])+str(args.pooling[3])
except:
    pooling_str = ""
if args.id:
    model_str = f"{args.model}_{args.id}_H{args.hidden}_h{args.seqlen}xw{width}{pooling_str}.pht"
else:
    model_str = f"{args.model}_H{args.hidden}_h{args.seqlen}xw{width}{pooling_str}.pht"
print("model_str", model_str)
log("model_str", model_str)

print("len data loader/total batches", len(train_loader))

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 1
train_losses = []
test_losses  = []
lowest_test_loss = 1e9

torch.autograd.set_detect_anomaly(True)

# Device configuration - use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_model(data_loader, model, loss_function, optimizer, epoch):
    num_batches = len(data_loader)
    total_loss = 0.0
    model.train()
    
    # Ensure model is on the correct device (CUDA if available)
    model.to(device)

    for X, y in tqdm(data_loader): #, desc=f"Epoch {epoch}/{args.epochs+epoch_start}"):
        X, y = X.to(device), y.to(device)

        outputs = model(X)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for X, y in tqdm(data_loader):
            outputs = model(X)
            total_loss += criterion(outputs, y).item()
            
    avg_loss = total_loss / num_batches
    return avg_loss

epoch_start = 0
ix_epoch = 0

if args.restart or not os.path.exists( model_str ):
    print("Starting from zero.")
    log("Starting from zero.")
    # Bootstrap values with an untrained network at "epoch 0".
    # Disadvantage is that this is usually large, not good for the plot.
    train_loss = test_model(train_loader, model, criterion)
    test_loss  = test_model(test_loader, model, criterion)
    train_losses.append( train_loss )
    test_losses.append( test_loss )
    print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
elif os.path.exists( model_str ):
    print(f"Continuing existing model {model_str}")
    log(f"Continuing existing model {model_str}")
    checkpoint = torch.load( model_str )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
    ix_epoch = epoch_start # if epochs=0, for eval
    train_losses = checkpoint['train_losses'],
    test_losses  = checkpoint['test_losses'],
    lowest_test_loss = checkpoint['lowest_test_loss']
    train_losses = train_losses[0] # these are tuples
    test_losses  = test_losses[0]  
    
if args.epochs > 0:
    for ix_epoch in range(epoch_start+1, epoch_start+args.epochs+1):
        print( f"Epoch {ix_epoch}/{args.epochs+epoch_start}" )
        train_loss = train_model(train_loader, model, criterion, optimizer, ix_epoch)
        test_loss  = test_model(test_loader, model, criterion)
        train_losses.append( train_loss )
        test_losses.append( test_loss )
        print(f"Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        log(f"Epoch {ix_epoch}/{args.epochs+epoch_start}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")
        print()

    torch.save({
            'epoch': ix_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lowest_test_loss':lowest_test_loss,
            'train_losses':train_losses,
            'test_losses':test_losses
        }, model_str)

'''
fv, lbl = next(iter(train_loader))
print("fv.shape", fv.shape)
print("lbl.shape", lbl.shape)
print(fv, fv.shape)
foo = model.predict(fv)
print(foo)
res_foo = oh_enc.inverse_transform(foo.detach().numpy())
res_lbl = oh_enc.inverse_transform(lbl.numpy())
for i, xy in enumerate(zip(res_foo, res_lbl)):
    x = xy[0]
    y = xy[1]
    a = ""
    if x != y:
        a = "ERR"
    print(i, x, y, a)
'''

if args.epochs > 0:
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    axs.plot( train_losses )
    axs.plot( test_losses )
    axs.set_ylim([0.0, None])
    fig.tight_layout()
    png_filename = model_str+".e"+str(ix_epoch)+".png"
    print( "Saving", png_filename )
    fig.savefig(png_filename, dpi=144)
    plt.pause(1.0)

fig, im = visualise_conv1(model)
plt.pause(1.0)
png_filename = model_str+".e"+str(ix_epoch)+"_conv1.pdf"
print( "Saving", png_filename )
fig.savefig(png_filename, dpi=144)

# ============================================================================
# Separate test file.
# ============================================================================

if args.testfile:
    log("Test file:", args.testfile)
    features, labels, oh_enc = load_data(args.testfile, args.seqlen, enc=oh_enc)
    print("Data/labels shape", features.shape, labels.shape)
    testing_data = TabSeparatedDataset(features, labels)
    test_loader = DataLoader(testing_data, batch_size=args.batchsize, shuffle=False)
    model.eval()
    predictions = []
    golds = []
    with torch.no_grad():
        for X, y in tqdm(test_loader):
            lbl = model.predict(X)
            # We get 2D Tensors, convert to numpy and loop over values.
            #for i, (pred, gold) in enumerate(zip(lbl.detach().numpy(), y.detach().numpy())):
            for i, (pred, gold) in enumerate(zip(lbl.cpu().numpy(), y.cpu().numpy())):
                res_lbl = oh_enc.inverse_transform(pred.reshape(1, -1))[0][0] # From [[0]] to 0.
                gold_lbl = oh_enc.inverse_transform(gold.reshape(1, -1))[0][0] 
                #if gold_lbl != 0:
                #    print(res_lbl, gold_lbl)
                predictions.append(res_lbl)
                golds.append(gold_lbl)
    #
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12,6), sharex=True, sharey=True)
    axs[0].vlines(range(len(predictions)), 0, predictions)
    axs[1].vlines(range(len(golds)), 0, predictions)
    axs[0].set_title( "Predictions" )
    axs[1].set_title( "Gold" )
    fig.suptitle( model_str )
    fig.tight_layout()
    png_filename = model_str+".e"+str(ix_epoch)+".testfile.png"
    print( "Saving", png_filename )
    fig.savefig(png_filename, dpi=144)

    cm = confusion_matrix(golds, predictions)
    print(cm)
    log(cm)
    cm = confusion_matrix(golds, predictions, normalize='true')
    with np.printoptions(precision=2, suppress=True):
        print(cm)
        log(cm)
    max_cm = cm.max()
    df_cm = pd.DataFrame(cm)
    # plt.figure(figsize=(10,7))
    scale = 1.0
    sn.set(font_scale=scale)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    fig.suptitle( model_str )
    #sn.heatmap( df_cm, annot=True, fmt='d', cmap='Blues', vmax=max_cm//10, ax=axs )
    #sn.heatmap( df_cm, annot=True, fmt='d', cmap='Blues', vmax=512, ax=axs )
    sn.heatmap( df_cm, annot=True, fmt='.2f', cmap='Blues', ax=axs )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    fig.tight_layout()
    png_filename = model_str+".e"+str(ix_epoch)+".cm.png"
    print( "Saving", png_filename )
    fig.savefig(png_filename, dpi=144)
    plt.pause(2.0)

# ============================================================================
# An unlabelled file is a file without targets. We can predict, but
# not calculate any statistics.
# ============================================================================

if args.unlabelled:
    log("Unlabelled file:", args.unlabelled)
    features = load_data(args.unlabelled, args.seqlen, getlabels=False)
    print("Data shape", features.shape)
    unlabelled_data = UnlabelledDataset(features)
    unlabelled_loader = DataLoader(unlabelled_data, batch_size=args.batchsize, shuffle=False) 
    model.eval()
    predictions = []
    with torch.no_grad():
        for X in tqdm(unlabelled_loader):
            lbl = model.predict(X)
            # We get 2D Tensors, convert to numpy and loop over values.
            for i, pred in enumerate(lbl.cpu().numpy()):
                res_lbl = oh_enc.inverse_transform(pred.reshape(1, -1))[0][0] # From [[0]] to 0.
                predictions.append(res_lbl)
    #print(predictions)
    # from torch_mocap_14.py
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    axs.vlines(range(len(predictions)), 0, predictions)
    axs.set_title( "Unlabelled data" )
    fig.suptitle( model_str )
    fig.tight_layout()
    png_filename = model_str+".e"+str(ix_epoch)+".unlabelled.png"
    print( "Saving", png_filename )
    fig.savefig(png_filename, dpi=144)
    #
    # Sequences?
    #predictions = [0,0,1,1,1,0,1,1,2,2,2,0,0]
    #               0   2   4 5 6 7 8  10
    in_seq = 0
    freq = 200 # 0.0050 # 200 Hz
    sequence = [0, 0, 0] # ?
    sequences = []
    for i, y in enumerate(predictions):
        if y == 0:
            if in_seq > 0:
                #print("END", i-1)
                sequence[2] = i*freq
                sequences.append(sequence)
                in_seq = 0
            continue
        if y != in_seq:
            if in_seq > 0:
                #print("END", i)
                sequence[2] = i*freq
                sequences.append(sequence)
            #print("START", i, y)
            in_seq = y
            sequence = [y, i*freq, 0]
    last_end = 0
    for s in sequences:
        st = ms_to_time(s[1])
        et = ms_to_time(s[2])
        du = s[2]-s[1]
        delta = s[1] - last_end
        last_end = s[2]
        s.append(delta) # For second run.
        #print(f"{s[0]} {st} - {et} {du:04} {delta:04}")
    '''
    new_sequences = [] # following is not correct FIXME
    for i in range(0, len(sequences)-1):
        if sequences[i][0] == sequences[i+1][0]: # next and this one same label
            if sequences[i+1][3] == 200: #with a gap of only one frame
                print("HICCUP", i, sequences[i+1]) # concatenate them
                new_sequences.append([sequences[i][0], sequences[i][1], sequences[i+1][2], 9999 ])
                i += 1
            else:
                new_sequences.append(sequences[i])
        else:
            new_sequences.append(sequences[i])
    print("")
    for s in new_sequences:
        st = ms_to_time(s[1])
        et = ms_to_time(s[2])
        du = s[2]-s[1]
        print(f"{s[0]} {st} - {et} {du:04}")
    '''

print( "END", time.asctime() )
print()
with open("./conv_04.log", "a") as f:
    f.write(datetime.datetime.now().strftime('END %Y%m%dT%H:%M:%S\n\n'))
plt.show()
