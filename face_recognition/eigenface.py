from sklearn.decomposition import PCA
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os.path

pic_set = []
averageface = 0
dir  = "male/"
orgshape = None
for rootdir,dirname,filenames in os.walk(dir):
	for filename in filenames:
		img = IMG.open(dir+filename).convert('L')
		p = np.array(img)
		orgshape = p.shape
		p = np.array(flatten(p.tolist()),float)
		averageface += p
		pic_set.append(p)

pic_set = np.array(pic_set)
print pic_set.shape
averageface = averageface / pic_set.shape[0]
for i in range(0, pic_set.shape[0]):
	pic_set[i] = pic_set[i] - averageface

L = np.dot(pic_set, pic_set.T)
lam, com = np.linalg.eig(L)
eigenface = np.array(com)

sumval = sum(lam)
temp = 0
i = 0
eigenface = np.dot(pic_set.T, eigenface).T #transpose to row matrix
while True:
	if (temp / sumval) <= 0.95:
		temp += lam[i]	
		i = i + 1
	else:
		break

eigenface = eigenface[0:i]

for i in range(eigenface.shape[0]):
	img = IMG.fromarray(eigenface[i].reshape(orgshape).astype(np.uint8))
	img.save("eigenface/eigenface"+str(i)+".jpg")

img = IMG.fromarray(averageface.reshape(orgshape).astype(np.uint8))
img.save('averageface.jpg')
