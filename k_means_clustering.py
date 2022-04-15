'''
This script quantize the images and uses kmeans clustering to train the model.

**********
Since there is no obvious elbow, I am using 10 as the elbow in the below script
**********
'''
import numpy as np
from skimage import data
from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
crocuses = io.imread("Crocuses_.jpg")
plt.figure()
plt.title("Crocuses Flowers")   
plt.imshow(crocuses)
plt.axis('off')
plt.show()
k = 2
r = 0
n = 0
crocuses = crocuses.reshape(331*500,3)
inertiaArr = np.array([0,0,0,0,0,0,0,0,0])
kArr = np.array([2,4,6,8,10,12,14,16,18])
while (k < 20):
    crocuses = io.imread("Crocuses_.jpg")
    crocuses = crocuses.reshape(331*500,3)
    flower = crocuses
    km_clustering = KMeans(n_clusters=k)
    cls_model = km_clustering.fit(flower)
    print("k =",k,", Inertia = ",cls_model.inertia_)
    inertiaArr[n] = cls_model.inertia_
    flower = flower.reshape(331,500,3)
    flower[:,:,r] = k*12
    if (r < 2):
        r += 1
    else:
        r -= 2
    plt.figure()
    plt.title("Crocuses Flowers")
    plt.imshow(flower)
    plt.axis('off')
    plt.show()
    k += 2
    n += 1
plt.figure()
plt.title("k VS Inertia")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.plot(kArr,inertiaArr)
plt.show()

#Lets take k = 10 for the Second task
rose = io.imread("Rose_.jpg")
plt.figure()
plt.title("Rose Flower")   
plt.imshow(rose)
plt.axis('off')
plt.show()
k = 10
rose[:,:,0] = k*12
plt.figure()
plt.title("Rose Flower")
plt.imshow(rose)
plt.axis('off')
plt.show()


# In[ ]:




