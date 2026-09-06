from PIL import Image
import pandas
import glob
import numpy as np
import h5py
import os


stride = 5             
numPix = 5             
width  = 160 // stride  
height = 160 // stride  

# File paths
trainingPaths = "./Aebs/carla_data/*.csv"  
saveFolder = './Aebs/data/'            
saveName = 'Downsampled.h5'        
#################################################

def getData(csv_path, verbose=True):
    if verbose:
        print("\nReading:", csv_path)

    data = pandas.read_csv(csv_path)
    
    y = np.array(data.distance_m).reshape(-1, 1).astype('float32')

    X = np.zeros([len(data), height, width]).astype('float32')
    img_folder = os.path.dirname(csv_path)

    for i, fn in enumerate(data.filename):
        img_path = os.path.join(img_folder, fn)
        img = Image.open(img_path)
        X[i] = computeTrainingImage(img)

        if verbose and i % 100 == 0:
            print(f"\tProcessed {i}/{len(data)} images")

    return X, y

def computeTrainingImage(img):
    img = np.array(img)

    img = np.array(Image.fromarray(img).convert('L').resize((160, 160))) / 255.0

    img2 = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            patch = img[stride*i:stride*(i+1), stride*j:stride*(j+1)].reshape(-1)
            img2[i, j] = np.mean(np.sort(patch)[-numPix:])

    img2 -= img2.mean()
    img2 += 0.5
    img2 = np.clip(img2, 0.0, 1.0)
    return img2.astype('float32')

X_train = np.zeros((0, height, width), dtype='float32')
y_train = np.zeros((0, 1), dtype='float32')

csv_paths = glob.glob(trainingPaths)

for csv_path in csv_paths:
    X, y = getData(csv_path)
    X_train = np.concatenate((X_train, X), axis=0)
    y_train = np.concatenate((y_train, y), axis=0)

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

with h5py.File(os.path.join(saveFolder, saveName), 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)

print("\n Data processing completed! Saved to:", os.path.join(saveFolder, saveName))
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
