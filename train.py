import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#load cifar 10 dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import pywt

import time

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download CIFAR-10
train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False) #True)
test_loader = DataLoader(test_dataset, batch_size=64)

import numpy as np

all_images = []
all_labels = []

for images, labels in train_loader:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

all_wavelets = []
#all_labels = []
for images, labels in train_loader:
    images_np = images.numpy()
    #labels_np = labels.numpy()

    #mask=np.random.rand(*images_np.shape) > 0.15
    #images_np=images_np*mask

    for img in images_np:
        # img shape: [C, H, W]
        wavelet_channels = []

        #C, H, W = img.shape
        #mask = np.random.rand(H, W) > 0.5
        #mask = mask[None,:, :]
        #img=img * mask

        for c in range(img.shape[0]):
            #LL, (LH, HL, HH) = pywt.dwt2(img[c], 'db4')
            coeffs = pywt.wavedec2(img[c],wavelet='haar',level=0) #4) 0
            #print(np.shape(np.stack([LL, LH, HL, HH])))
            arr, slices = pywt.coeffs_to_array(coeffs)
            #print(np.shape(arr))
            #wavelet_channels.append(np.stack([LL, LH, HL, HH]))
            wavelet_channels.append(arr.flatten())

        # shape: [C, 4, H/2, W/2]
        all_wavelets.append(np.stack(wavelet_channels))

    #all_labels.extend(labels_np)

all_images = np.array(all_wavelets)
#all_labels = np.array(all_labels)

np.random.seed(1986)

# Concatenate batches
X = np.concatenate(all_images, axis=0)
y = np.concatenate(all_labels, axis=0)

print(X.shape)
#print(X)

#X_noise=X.copy()
##X_noise[:,:,16:,:]=0.5
#X_noise[:,:,::4,::4]=0 #X_noise[:,:,:,:] +0.4*np.random.randn(*X_noise.shape) #0
#X_noise[:,:,1::4,::4]=0
#X_noise[:,:,::4,1::4]=0s
#X_noise[:,:,1::4,1::4]=0
##for i in range(0,16):
##  X_noise[i::16,0:2,i,16]=0
##
#X_noise=X_noise.reshape((50000,3*32*32))
#X_noise=X_noise.T

X=X.reshape((50000,3*32*32))
X=X.T

print(X.shape)
print(y.shape)

#idx = np.argsort(y)
#X = X[:,idx]
#y = y[idx]
X=X[:,y==1]
#X=X[:,0:2]
#X=X[:,:150]
#X=np.concatenate((X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X),axis=1)

all_images = []
all_labels = []

for images, labels in test_loader:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

all_wavelets = []
#all_labels = []
for images, labels in test_loader:
    images_np = images.numpy()
    #labels_np = labels.numpy()

    for img in images_np:
        # img shape: [C, H, W]
        wavelet_channels = []

        for c in range(img.shape[0]):
            #LL, (LH, HL, HH) = pywt.dwt2(img[c], 'db4')
            #print(np.shape(np.stack([LL, LH, HL, HH])))
            coeffs = pywt.wavedec2(img[c],wavelet='haar',level=0) #4) 0
            arr, slices = pywt.coeffs_to_array(coeffs)
            #print(np.shape(arr))
            #wavelet_channels.append(np.stack([LL, LH, HL, HH]))
            wavelet_channels.append(arr.flatten())

        # shape: [C, 4, H/2, W/2]
        all_wavelets.append(np.stack(wavelet_channels))

    #all_labels.extend(labels_np)

all_images = np.array(all_wavelets)
#all_labels = np.array(all_labels)

# Concatenate batches
X_test = np.concatenate(all_images, axis=0)
y_test = np.concatenate(all_labels, axis=0)

print(X_test.shape)

X_test=X_test.reshape((10000,3*32*32))
X_test=X_test.T

#X = X.astype(np.float16)
#X_noise = X_noise.astype(np.float16)
#X_test = X_test.astype(np.float16)

del train_loader
del test_loader
del all_images
del all_labels

scaler_wavelet = StandardScaler()
X2 = scaler_wavelet.fit_transform(X.T).T
X2=np.concatenate((1*np.maximum(X2,0),1*np.maximum(-(X2),0)),axis=0) #128 #-0.5
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
X=X2.copy().astype(np.float32)

#X2=np.concatenate((np.maximum(X_noise-128,0),np.maximum(-(X_noise-128),0)),axis=0)
##X2=np.concatenate((np.maximum(X2-64,0),np.maximum(-(X2-64),0)),axis=0)
#X_noise=X2.copy().astype(np.float32)

print(X.shape)

X2 = scaler_wavelet.transform(X_test.T).T
X2=np.concatenate((1*np.maximum(X2,0),1*np.maximum(-(X2),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
#X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
X2=np.concatenate((2*np.maximum(X2-0.5,0),2*np.maximum(-(X2-0.5),0)),axis=0)
X_test=X2.copy().astype(np.float32)

del X2

#######

import gc
gc.collect()


start = time.perf_counter()

# Perform PCA
scaler_list=[]
pca_list=[]
X_pca_list=[]
max_iter=20 #40 #4 #2 #20 #10
for i in range(0,max_iter):
  print(i)
  scaler = StandardScaler()
  if i==0:
    X_norm = scaler.fit_transform(X.T)
    #X_norm=X.T
  else:
    X_norm = scaler.fit_transform(X_pca_list[i-1].T)
    #X_norm=X_norm*(1/X_norm.shape[1])*(np.arange(X_norm.shape[1],X_norm.shape[1]*2,1))
    #X_norm=X_pca_list[i-1].T
  X_norm=X_norm.T
  #X_norm=X_norm-np.mean(X_norm,axis=0)
  #X_norm=X_norm/np.std(X_norm,axis=0)
  #exit(0)
  #print(X)
  #print(np.mean(X_norm,axis=0))
  m1_1=0#np.mean(np.maximum(X_norm,0))
  m1_2=0#np.mean(np.maximum(-X_norm,0))
  X_div=np.concatenate((np.maximum(X_norm,0)-m1_1,np.maximum(-X_norm,0)-m1_2),axis=0)
  #print(np.shape(X_div))
  #print(X_div)
  #m1=np.mean(X_div,axis=0)
  #X_div=X_div-m1
  #temp=np.matmul(X_div.T,X_div)
  #plt.figure()
  #plt.imshow(temp, cmap='viridis')   # apply colormap
  #plt.colorbar()
  #plt.show()
  #X_div=X_div+np.random.normal(0,0.35,X_div.shape)#np.random.rand(*X_div.shape)
  if i==0: #max_iter-1:
    n_components = 2000 #150 #500 #4000 #200 #500
    #pca = PCA(n_components=n_components, copy=False, svd_solver='full')
  else:
    n_components = 2000 #150 #500 #4000 #int(n_components*1.05) #500 #200 #+500*(i)  # Set number of principal components
    #pca=SparsePCA(n_components=n_components)
  pca = PCA(n_components=n_components, copy=False, svd_solver='full')
  #pca=SparsePCA(n_components=n_components)
  X_pca = pca.fit_transform(X_div.T)
  #X_pca[X_pca<0]=-np.sqrt(np.abs(X_pca[X_pca<0]))
  #X_pca[X_pca>=0]=np.sqrt(np.abs(X_pca[X_pca>=0]))
  X_pca=X_pca.T
  #print(np.max(X_pca,0).shape)

  scaler_list.append(scaler)
  pca_list.append(pca)
  X_pca_list.append(X_pca)

  #X_pca3=np.multiply(X_pca3, np.random.rand(*X_pca3.shape) )


end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")
#######

# Perform PCA
max_iter=20 #40 #4 #2 #20 #10
for i in range(0,max_iter):
  print(i)
  #scaler = StandardScaler()
  if i==0:
    X_norm_test = scaler_list[i].transform(X_test.T)
    #X_norm_test=X_test.T
    #print(np.shape(X_test))
  else:
    X_norm_test = scaler_list[i].transform(X_pca_test.T)
    #X_norm_test=X_norm_test*(1/X_norm_test.shape[1])*(np.arange(X_norm_test.shape[1],X_norm_test.shape[1]*2,1))
    #X_norm_test=X_pca_test.T
  X_norm_test=X_norm_test.T
  #X_norm=X_norm-np.mean(X_norm,axis=0)
  #X_norm=X_norm/np.std(X_norm,axis=0)
  #exit(0)
  #print(X)
  #print(np.mean(X_norm_test,axis=0))
  m1_1=0#np.mean(np.maximum(X_norm,0))
  m1_2=0#np.mean(np.maximum(-X_norm,0))
  X_div=np.concatenate((np.maximum(X_norm_test,0)-m1_1,np.maximum(-X_norm_test,0)-m1_2),axis=0)
  #print(X_div)
  #m1=np.mean(X_div,axis=0)
  #X_div=X_div-m1
  #temp=np.matmul(X_div.T,X_div)
  #plt.figure()
  #plt.imshow(temp, cmap='viridis')   # apply colormap
  #plt.colorbar()
  #plt.show()
  #X_div=X_div+np.random.normal(0,0.35,X_div.shape)#np.random.rand(*X_div.shape)
  if i==max_iter-1:
    n_components =500 #500
  else:
    n_components = 500 #*(i+1)  # Set number of principal components
  #pca = PCA(n_components=n_components)
  X_pca_test = pca_list[i].transform(X_div.T)
  #X_pca[X_pca<0]=-np.sqrt(np.abs(X_pca[X_pca<0]))
  #X_pca[X_pca>=0]=np.sqrt(np.abs(X_pca[X_pca>=0]))
  X_pca_test=X_pca_test.T
  #print(np.max(X_pca,0).shape)

  #scaler_list.append(scaler)
  #pca_list.append(pca)
  #X_pca_list.append(X_pca)

######
#'''
#def quantize(X, bits=4):
#    levels = 2**bits - 1

#    xmin = X.min()
#    xmax = X.max()

#    scale = (xmax - xmin) / levels

#    X_q = np.round((X - xmin) / scale)
#    X_q=X_q*scale+xmin

#    return X_q, scale, xmin

def quantize(X, bits=4, axis=None):
    """
    Quantizes a matrix X.
    
    Parameters:
    - X: Input NumPy array.
    - bits: Number of quantization bits.
    - axis: None for global quantization, 
            0 for column-by-column, 
            1 for row-by-row.
    """
    levels = 2**bits - 1
    
    # Calculate min and max along the specified axis
    # keepdims=True ensures proper broadcasting shapes
    xmin = X.min(axis=axis, keepdims=True)
    xmax = X.max(axis=axis, keepdims=True)
    
    # Handle potential division by zero if max == min
    scale = (xmax - xmin) / levels
    scale = np.where(scale == 0, 1e-8, scale) 
    
    # Quantize and dequantize
    X_q = np.round((X - xmin) / scale)
    X_q = X_q * scale + xmin
    
    return X_q, scale, xmin

from scipy.spatial.distance import cdist
def sort_cols_by_l2(matrix,X):
    #distances = np.linalg.norm(matrix - matrix[:,235], axis=0) #compare with 235 row, not 0 row
    #indices = np.argsort(distances)
    #return matrix[:,indices]

    # L2 distance of each column from the first column
    #distances = np.linalg.norm(
    #  matrix - matrix[:, [999]], axis=0
    #)
    
    # Cosine distance from the first column
    distances = cdist(
      matrix.T,
      matrix[:, 2316].reshape(1, -1), #2316 #999 #99 #9 #1234 #123 #789 #111
      metric='cosine'
    ).flatten()

    # Sort columns by distance
    sorted_matrix = X[:,np.argsort(distances)]
    sorted_matrix2=matrix[:,np.argsort(distances)]
    dist2=np.sort(distances)
    print(dist2[0:20])
    return sorted_matrix,sorted_matrix2

scaler = StandardScaler()
X_norm4 = X_pca #scaler4.fit_transform(X_pca3)
#X_norm2=X_norm2-np.mean(X_norm2,axis=0)
#X_norm2=X_norm2/np.std(X_norm2,axis=0)
m4_1=0#np.mean(np.maximum(X_norm4,0))
m4_2=0#np.mean(np.maximum(-X_norm4,0))
X_div4=np.concatenate((np.maximum(X_norm4,0)-m4_1,np.maximum(-X_norm4,0)-m4_2),axis=0)

#X_div4=1*np.random.random(X_div4.shape)
X_div4=1*np.maximum(np.random.normal(0,1,size=X_div4.shape),0)
#X_pca=X_div4[:(X_div4.shape[0]//2),:]-X_div4[(X_div4.shape[0]//2):,:]
##X_pca=1*(np.random.normal(0,1,size=X_pca.shape)) #0,1 #generate random latent vectors
#X_pca, scale, xmin = quantize(X_pca, bits=1)
#X_pca[X_pca.shape[0]//2:,:]=0 #mask out half the latent vector dimension
#X_pca[(X_pca.shape[0]//4):]=0

#print(X.shape)
X,X_pca=sort_cols_by_l2(X_pca,X)#rank the X_pca vectors to the first vector
#print(X.shape)
#print(X_pca.shape)
#exit(0)

X_norm4_test = X_pca_test #scaler4.fit_transform(X_pca3)
#X_norm2=X_norm2-np.mean(X_norm2,axis=0)
#X_norm2=X_norm2/np.std(X_norm2,axis=0)
m4_1=0#np.mean(np.maximum(X_norm4,0))
m4_2=0#np.mean(np.maximum(-X_norm4,0))
X_div4_test=np.concatenate((np.maximum(X_norm4_test,0)-m4_1,np.maximum(-X_norm4_test,0)-m4_2),axis=0)

X_div4_test=1*np.random.random(X_div4_test.shape)
X_pca_test=X_div4_test[:(X_div4_test.shape[0]//2),:]-X_div4_test[(X_div4_test.shape[0]//2):,:]
#'''

'''
#X_div4=X_div4+np.random.rand(*X_div4.shape)
print(X_pca.shape)

dataset_size=X_pca.shape[1] #50000

print(X_pca.shape)
rr=np.random.randint(0,dataset_size,2)
#rr=[18561,39880]

print(rr[0])
print(rr[1])
print(y[rr[0]])
print(y[rr[1]])
#print(a)
#print(b)

for i3 in range(0,dataset_size-30,20):
    a=X_div4[:,i3].copy()
    b=X_div4[:,i3+20].copy()
    a_norm = np.linalg.norm(a, axis=0)
    b_norm = np.linalg.norm(b, axis=0)
    for i2 in range(0,20):#50000
        i=i3+i2
        X_div4[:,i]=1*a*(1-i2/20)+1*b*(i2/20)
        norm = np.linalg.norm(X_div4[:,i] , axis=0)
        X_div4[:,i]=X_div4[:,i]/norm*(a_norm*(1-i2/20)+b_norm*(i2/20))

X_div4=X_div4[:(X_div4.shape[0]//2),:]-X_div4[(X_div4.shape[0]//2):,:]+m4_1+m4_2
X_pca =X_div4 # scaler4.inverse_transform(X_div4)
'''

'''
scaler = StandardScaler()
X_norm4 = X_pca #scaler4.fit_transform(X_pca3)
#X_norm2=X_norm2-np.mean(X_norm2,axis=0)
#X_norm2=X_norm2/np.std(X_norm2,axis=0)
m4_1=0#np.mean(np.maximum(X_norm4,0))
m4_2=0#np.mean(np.maximum(-X_norm4,0))
X_div4_temp=np.concatenate((np.maximum(X_norm4,0)-m4_1,np.maximum(-X_norm4,0)-m4_2),axis=0)

X_div4=X_div4_temp.copy()
X_div4[::2,:]=X_div4[:X_div4.shape[0]//2,:]
X_div4[1::2,:]=X_div4[X_div4.shape[0]//2:,:]
#X_div4=X_div4[:(X_div4.shape[0]//2),:]-X_div4[(X_div4.shape[0]//2):,:]+m4_1+m4_2
print(np.shape(X_div4))

scaler = StandardScaler()
X_norm4_test = X_pca_test #scaler4.fit_transform(X_pca3)
##X_norm2=X_norm2-np.mean(X_norm2,axis=0)
##X_norm2=X_norm2/np.std(X_norm2,axis=0)
m4_1=0#np.mean(np.maximum(X_norm4,0))
m4_2=0#np.mean(np.maximum(-X_norm4,0))
X_div4_temp_test=np.concatenate((np.maximum(X_norm4_test,0)-m4_1,np.maximum(-X_norm4_test,0)-m4_2),axis=0)

X_div4_test=X_div4_temp_test.copy()
X_div4_test[::2,:]=X_div4_test[:X_div4_test.shape[0]//2,:]
X_div4_test[1::2,:]=X_div4_test[X_div4_test.shape[0]//2:,:]
##X_div4=X_div4[:(X_div4.shape[0]//2),:]-X_div4[(X_div4.shape[0]//2):,:]+m4_1+m4_2
print(np.shape(X_div4_test))
#'''

######
total_weights_count=0

X_recon_list=[]
for i in range(max_iter-1,-1,-1):
  if i!=(max_iter-1):
    X_pca_temp=X_recon3
  if i==(max_iter-1):
    X_pca_temp=X_pca
  #X_pca_temp[X_pca_temp<0]=-np.square(np.abs(X_pca_temp[X_pca_temp<0]))
  #X_pca_temp[X_pca_temp>=0]=np.square(np.abs(X_pca_temp[X_pca_temp>=0]))
  #if i==(max_iter-1):
  #if i!=0:
  #  pca_list[i].components_[:,:]=pca_list[i].components_[:,:]+1 #*np.concatenate((np.eye(100),np.eye(100)),axis=1)

  total_weights_count=total_weights_count+len((pca_list[i].components_).flatten())

  #pca_list[i].components_ , scale, xmin= quantize(pca_list[i].components_, bits=2, axis=0) #quantize pca weights
  X_recon3 = pca_list[i].inverse_transform(X_pca_temp.T)
  #else:
  #X_recon3 = pca_list[i].inverse_transform(X_recon3.T)
  X_recon3=X_recon3.T
  #X_recon3=X_recon3+m3
  #print(X_recon3.shape[0]//2)
  X_recon3=X_recon3[:(X_recon3.shape[0]//2),:]-X_recon3[(X_recon3.shape[0]//2):,:]#+m3_1-m3_2
  #if i>0:
  #  X_recon3=X_recon3.T
  #  X_recon3=X_recon3/(1/X_recon3.shape[1])/(np.arange(X_recon3.shape[1],X_recon3.shape[1]*2,1))
  #  X_recon3=X_recon3.T
  X_recon3 = scaler_list[i].inverse_transform(X_recon3.T)
  X_recon3=X_recon3.T

  X_recon_list.append(X_recon3)

X_recon=X_recon3
#print("PCA components shape:", pca.components_.shape)
#print("Transformed data shape:", X_pca.shape)

print('total_weights_count ',total_weights_count)
#####
'''
print(np.sum(np.abs(X_div4)<0.00001))
print(np.shape(X_div4))

for i in range(0,max_iter):
  print(np.sum(np.abs(pca_list[i].components_)<0.00001))
  print(np.shape(pca_list[i].components_))
  plt.figure()
  plt.hist(((pca_list[i].components_.flatten())),bins=1000)
plt.show()
'''
#####

#exit(0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

'''
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#scalersvm = StandardScaler()
#X_svm = scalersvm.fit_transform(X_pca)

# Compute L2 norm of each column
norms = np.linalg.norm(X_div4, axis=0, keepdims=True)
print(X.shape)
# Avoid division by zero
norms[norms == 0] = 0
# Normalize columns
X_svm = np.array(X_div4) / norms #np.concatenate((np.array(X_div4) / norms,np.array(X_div4)),axis=0)

print(X_svm.shape)

#scalersvm = StandardScaler()
#X_svm = scalersvm.fit_transform(X_pca)

# Compute L2 norm of each column
norms_test = np.linalg.norm(X_div4_test, axis=0, keepdims=True)
print(X_test.shape)
# Avoid division by zero
norms_test[norms_test == 0] = 0
# Normalize columns
X_svm_test = np.array(X_div4_test) / norms_test #np.concatenate((np.array(X_div4_test) / norms_test,np.array(X_div4_test)),axis=0)

print(X_svm.shape)

"""
# Multiclass logistic regression
model = LogisticRegression(
    #multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_svm.T, y)

# Predict
y_pred = model.predict(X_svm_test.T)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""

#''
summ=0
sum_margin=0
for i in range(0,10):
    model = SVC(kernel='poly', degree=2) #SVC(kernel='rbf',C=1.0,gamma='scale')  #(kernel='sigmoid', gamma='scale', coef0=0.0) #(kernel='poly', degree=2) #(kernel='linear',C=1)

    #n=10*np.random.rand(*X_svm[:,::100].shape)

    # Train
    #model.fit((X_svm[:,::100]+n).T, y[::100]==i)
    model.fit(X_svm[:,::5].T, y[::5]==i)

    # Get weight vector
    ##w = model.coef_[0]

    # Compute margin
    ##margin = 2 / np.linalg.norm(w)

    ##print("Margin:", margin)
    ##sum_margin=sum_margin+margin

    # Predict
    y_pred = model.predict((X_svm_test[:,0::1]).T)

    # Evaluate
    acc=accuracy_score(y_test[0::1]==i , y_pred)
    summ=summ+acc
    print("Accuracy:", acc)
print(summ/10)
print(sum_margin/10)
#'''
#####

import gc
gc.collect()

#X=X_noise

res=32

temp_out=0.5*X_recon[:(X_recon.shape[0]//2),:]-0.5*X_recon[(X_recon.shape[0]//2):,:]+0.5
#temp_out=0.5*temp_out[:(temp_out.shape[0]//2),:]-0.5*temp_out[(temp_out.shape[0]//2):,:]+0.5
#temp_out=0.5*temp_out[:(temp_out.shape[0]//2),:]-0.5*temp_out[(temp_out.shape[0]//2):,:]+0.5
#temp_out=0.5*temp_out[:(temp_out.shape[0]//2),:]-0.5*temp_out[(temp_out.shape[0]//2):,:]+0.5
#temp_out=0.5*temp_out[:(temp_out.shape[0]//2),:]-0.5*temp_out[(temp_out.shape[0]//2):,:]+0.5
temp_out=1*temp_out[:(temp_out.shape[0]//2),:]-1*temp_out[(temp_out.shape[0]//2):,:]
temp_out=scaler_wavelet.inverse_transform(temp_out.T).T

temp_X=0.5*X[:(X.shape[0]//2),:]-0.5*X[(X.shape[0]//2):,:]+0.5
#temp_X=0.5*temp_X[:(temp_X.shape[0]//2),:]-0.5*temp_X[(temp_X.shape[0]//2):,:]+0.5
#temp_X=0.5*temp_X[:(temp_X.shape[0]//2),:]-0.5*temp_X[(temp_X.shape[0]//2):,:]+0.5
#temp_X=0.5*temp_X[:(temp_X.shape[0]//2),:]-0.5*temp_X[(temp_X.shape[0]//2):,:]+0.5
#temp_X=0.5*temp_X[:(temp_X.shape[0]//2),:]-0.5*temp_X[(temp_X.shape[0]//2):,:]+0.5
temp_X=1*temp_X[:(temp_X.shape[0]//2),:]-1*temp_X[(temp_X.shape[0]//2):,:]
temp_X=scaler_wavelet.inverse_transform(temp_X.T).T
print(temp_X.shape)

out_map=np.zeros((32,32*20,3))
out_map2=np.zeros((32,32*20,3))

i2= 0 #204 #np.random.randint(0,(50000-30)//20)#2162
print(i2)
print((y[i2*20],y[i2*20+20]))
for i3 in range(0+i2*20,i2*20+20):
    i=int((i3-i2*20))
    rand_idx=i3 #i3%2 #i3
    #rand_idx = np.random.randint(0, X.shape[0])
    #rand_idx=0
    print("Random index:", rand_idx)
    #plt.figure()
    #plt.imshow(X_recon2[0].reshape(128, 128), cmap='gray')
    #plt.figure()
    #print(X_recon[:,rand_idx].shape)
    #plt.imshow(X_recon[:,rand_idx].reshape(res, res), cmap='viridis')
    #temp_out[:,rand_idx].reshape(3,res,res)
    temp2=np.zeros((3,32,32))
    for j in range(0,3):
      temp=temp_out[:,rand_idx].reshape(3,32,32)
      #temp=(np.squeeze(temp[j,0,:,:]),(np.squeeze(temp[j,1,:,:]),np.squeeze(temp[j,2,:,:]),np.squeeze(temp[j,3,:,:])))
      #temp2[j,:,:]=pywt.idwt2(temp, 'db4')
      temp=temp[j,:,:]
      coeffs = pywt.array_to_coeffs(temp, slices, output_format='wavedec2')
      temp2[j,:,:] = pywt.waverec2(coeffs, 'haar')
    out_map[:,i*res:(i+1)*res,:]=temp2.transpose(1, 2, 0) #X_recon
    for j in range(0,3):
      temp=temp_X[:,rand_idx].reshape(3,32,32)
      #temp=(np.squeeze(temp[j,0,:,:]),(np.squeeze(temp[j,1,:,:]),np.squeeze(temp[j,2,:,:]),np.squeeze(temp[j,3,:,:])))
      #temp2[j,:,:]=pywt.idwt2(temp, 'db4')
      temp=temp[j,:,:]
      coeffs = pywt.array_to_coeffs(temp, slices, output_format='wavedec2')
      temp2[j,:,:] = pywt.waverec2(coeffs, 'haar')
    out_map2[:,i*res:(i+1)*res,:]=temp2.transpose(1,2,0)
    #out_map2[:,i*res:(i+1)*res,:]=temp_X[:,rand_idx].reshape(3,res,res).transpose(1,2,0)

    #plt.figure()
    #plt.imshow(temp_out[:,rand_idx].reshape(3,res,res).transpose(1,2,0))
    #plt.imshow(X_recon_one[:,rand_idx].reshape(res, res), cmap='gray')
    #plt.figure()
    #print(np.array(X)[rand_idx].shape)
    #plt.imshow(np.array(X)[:,rand_idx].reshape(res, res), cmap='gray')
plt.figure(figsize=(20,4)) #10*32,32*2) )
from skimage.transform import resize
#resized = resize(out_map, (28*4, 28*20*4), anti_aliasing=True)
out_map=np.concatenate((out_map[:,:(10*32),:],out_map[:,(10*32):,:]),axis=0)
plt.imshow(out_map/255)#, cmap='viridis')
#plt.show()
plt.figure(figsize=(20,4))
out_map2=np.concatenate((out_map,out_map2[:,:(10*32),:],out_map2[:,(10*32):,:]),axis=0)
plt.imshow(out_map2) #, cmap='viridis')

plt.figure()
#plt.imshow(temp_X[:,i2*20].reshape(3,res,res).transpose(1, 2, 0))
plt.figure()
#plt.imshow(temp_X[:,i2*20+20].reshape(3,res,res).transpose(1, 2, 0))
plt.show()
#exit(0)

#print(temp_X[:,i2*20+20].reshape(3,res,res).transpose(1, 2, 0))