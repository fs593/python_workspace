from sklearn.decomposition import PCA
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os.path
import copy
def pca_decompo(sample_mat, ncomponent):
	average_sam = 0
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
	return (eigen, low_dim_sample)
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

print pic_set.shape
averageface = averageface / pic_set.shape[0]
for i in range(0, pic_set.shape[0]):
	pic_set[i] = pic_set[i] - averageface

pic_setO = copy.deepcopy(pic_set)
L = np.dot(pic_set, pic_set.T)
print "L shape:"+str(L.shape)
lam, com = np.linalg.eig(L)
_eigenface = np.array(com)
print _eigenface.shape

sumval = sum(lam)
temp = 0
i = 0
eigenface = np.dot(pic_set.T, _eigenface).T #transpose to row matrix
print eigenface.shape

#while True:
#	if (temp / sumval) <= 0.95:
#		temp += lam[i]	
#		i = i + 1
#	else:
#		break
i = 75
eigenface = eigenface[0:i]
pic_set_lo = np.dot(pic_setO, eigenface.T)
print pic_set_lo.shape
#normalize eigenface
for i in range(0, eigenface.shape[0]):
	eigenface[i] = eigenface[i] / np.linalg.norm(eigenface[i], 2)

test_face = []
test_sample_dir1= "test_sample1/"
for rootdir, dirname, filenames in os.walk(test_sample_dir1):
	for filename in filenames:
		img = IMG.open(test_sample_dir1 + filename).convert('L')
		p = np.array(img)
		p = np.array(flatten(p.tolist()), float) - averageface
		test_face.append(p)

test_face = np.array(test_face) #all original test case
img_match = IMG.open("match1.jpg").convert('L')
match_vec = np.array(img_match)
match_vec = np.array(flatten(match_vec.tolist()), float)# original matching face
match_vec = match_vec - averageface
#print eigenface.shape
#print test_face.shape
decision_mat = np.dot(eigenface, test_face.T).T #test case after dimentionality reduction
decision_vec = np.dot(eigenface, match_vec.T) #matching face after dimentionality reduction
#print decision_mat.shape
#print decision_vec.shape
rank = np.dot(decision_mat, decision_vec)

print rank


#for i in range(eigenface.shape[0]):
#	img = IMG.fromarray(eigenface[i].reshape(orgshape).astype(np.uint8))
#	img.save("eigenface/eigenface"+str(i)+".jpg")

#img = IMG.fromarray(averageface.reshape(orgshape).astype(np.uint8))
#img.save('averageface.jpg')
