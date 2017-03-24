from sklearn.decomposition import PCA
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os.path
import copy
#using the PCA to reduce the dimention of Training set
#for each pic in original Test set, project this test pic in to the dimention reduced training set
def pca_decompo(sample_mat, ncomponent):
	average_sam = 0
	sample_mat = np.array(sample_mat)
	for i in range(0,sample_mat.shape[0]):
		average_sam = average_sam + sample_mat[i]

	average_sam = average_sam / sample_mat.shape[0]
	for i in range(0, sample_mat.shape[0]):
		sample_mat[i] = sample_mat[i] - average_sam

	L = np.dot(sample_mat, sample_mat.T)
	vari, com = np.linalg.eig(L)
	_eigenface = np.array(com)
	eigenface = np.dot(sample_mat.T, _eigenface).T
	eigenface = eigenface[0:ncomponent]
	sample_mat_ld = np.dot(sample_mat, eigenface.T)
	return (eigenface, sample_mat_ld)
	#eigen: largest $ncomponent$ eigen vector
	#low_dim_sample: dimentional reduced samples

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
pic_setO = copy.deepcopy(pic_set)

averageface = averageface / pic_set.shape[0]
for i in range(0, pic_set.shape[0]):
	pic_set[i] = pic_set[i] - averageface

L = np.dot(pic_set, pic_set.T)

lam, com = np.linalg.eig(L)
_eigenface = np.array(com)


sumval = sum(lam)
temp = 0
i = 0
eigenface = np.dot(pic_set.T, _eigenface).T #transpose to row matrix

#while True:
#	if (temp / sumval) <= 0.95:
#		temp += lam[i]	
#		i = i + 1
#	else:
#		break
ncomponent = 90
eigenface = eigenface[0:ncomponent]
pic_set_ld = np.dot(pic_set, eigenface.T)

#normalize eigenface
for i in range(0, eigenface.shape[0]):
	eigenface[i] = eigenface[i] / np.linalg.norm(eigenface[i], 2)

test_face = []
test_sample_dir1= "test_sample1/"
for rootdir, dirname, filenames in os.walk(test_sample_dir1):
	for filename in filenames:
		img = IMG.open(test_sample_dir1 + filename).convert('L')
		p = np.array(img)
		p = np.array(flatten(p.tolist()), float)
		test_face.append(p)

test_face = np.array(test_face)
def dist(u,v):
	u = np.array(u)
	u = u / np.linalg.norm(u,2)
	v = np.array(v)
	v = v / np.linalg.norm(v,2)
	#return np.linalg.norm(u-v, 2)
	return np.dot(u,v)

result = []
for i in range(0, test_face.shape[0]):
	maxdist = -float('inf')
	maxj = 0
	match_pic = test_face[i] - averageface
	match_pic_ld = np.dot(match_pic, eigenface.T)
	for j in range(0, pic_set_ld.shape[0]):
		t = dist(match_pic_ld, pic_set_ld[j])
		if t > maxdist:
			maxdist = t
			maxj = j
	result.append(maxj)

for i in range(0, len(result)):
	img = IMG.fromarray(test_face[i].reshape(orgshape).astype(np.uint8))
	img.save("result/"+str(i)+"test.jpg")
	img = IMG.fromarray(pic_setO[result[i]].reshape(orgshape).astype(np.uint8))
	img.save("result/"+str(i)+"match.jpg")

#for i in range(0,test_eigen.shape[0]):
#	img = IMG.fromarray(test_eigen[i].reshape(orgshape).astype(np.uint8))
#	img.save("test_eigen/test_eigen"+str(i)+".jpg")


#for i in range(eigenface.shape[0]):
#	img = IMG.fromarray(eigenface[i].reshape(orgshape).astype(np.uint8))
#	img.save("eigenface/eigenface"+str(i)+".jpg")

#img = IMG.fromarray(averageface.reshape(orgshape).astype(np.uint8))
#img.save('averageface.jpg')
