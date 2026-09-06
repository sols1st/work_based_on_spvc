import h5py
import matplotlib.pyplot as plt

h5_path = './Aebs/data/Downsampled.h5'

with h5py.File(h5_path, 'r') as f:
    X_train = f['X_train']  
    y_train = f['y_train']  

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    img = X_train[50]
    label = y_train[50][0]
    
    plt.imshow(img, cmap='gray')
    plt.title(f"Distance: {label:.2f} m")
    plt.axis('off')
    plt.show()
