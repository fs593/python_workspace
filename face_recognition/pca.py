from sklearn.decomposition import PCA
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os
pic_set = []
averageface = 0
picshape = None
for i in range(1,3):
	p = np.array(IMG.open('ekavaz/ekavaz.'+str(i)+'.jpg').convert('L'))
	picshape = p.shape
	p = np.array(flatten(p.tolist()),float)
	averageface += p
	pic_set.append(p)
#filelist = os.listdir('gotone/')
#for i in range(0, len(filelist)):
#	p = np.array(IMG.open('gotone/'+filelist[i]).convert('L'))
#	picshape = p.shape
#	p = np.array(flatten(p.tolist()),float)
#	averageface += p
#	pic_set.append(p)

pic_set = np.array(pic_set)
averageface = averageface / pic_set.shape[0]

for i in range(0, pic_set.shape[0]):
	pic_set[i] = pic_set[i] - averageface

#pca = PCA(n_components = 0.95, copy = True)
#pca.fit(pic_set)

#com = pca.components_
#vari= pca.explained_variance_

covMatrix = np.dot(pic_set, pic_set.T)

vari , com= np.linalg.eig(covMatrix)

com = np.dot(pic_set.T, com).T

eighenface =com[0]
im = IMG.fromarray(eighenface.reshape(picshape).astype(np.uint8))
im.save('eighenface.png')

im = IMG.fromarray(averageface.reshape(picshape).astype(np.uint8))
im.save('averageface.png')