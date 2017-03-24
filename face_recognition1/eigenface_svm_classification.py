from sklearn import svm
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os.path
import copy
#each person have numbers of face pic, using PCA to reduce the dimention of pic set
#in feature sub space, we use SVM to build a classifier of each person's face in sub feature space
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
pic_sign = []
averageface = 0
dir  = "male/"
orgshape = None
dir_list = []
for dirs in os.listdir(dir):
	dir_list.append(dirs)

for i in range(0, len(dir_list)):
	for rootdir, dirs, filenames in os.walk(dir+dir_list[i]):
		for j in range(1, 11):
			img = IMG.open(dir+dir_list[i]+"/"+dir_list[i]+"."+str(j)+".jpg").convert('L')
			p = np.array(img)
			orgshape = p.shape
			p = np.array(flatten(p.tolist()),float)
			averageface += p
			pic_set.append(p)# add pic to the training set
			pic_sign.append(i)# labeling the data by using their id

pic_set = np.array(pic_set)
pic_seto = copy.deepcopy(pic_set)
print pic_set.shape
averageface = averageface / pic_set.shape[0]

for i in range(0, pic_set.shape[0]):
	pic_set[i] -= averageface

ncomponent = 80
eigen, pic_set_ld = pca_decompo(pic_set, ncomponent)
clf = svm.SVC(kernel = 'rbf', decision_function_shape='ovo')
clf.fit(pic_set_ld, pic_sign)

#get test samples
test_pic = []
test_dir = "test_sample/"
for rootdir, dirs, filenames in os.walk(test_dir):
	for file in filenames:
		img = IMG.open(test_dir + file).convert('L')
		p = np.array(img)
		p = np.array(flatten(p.tolist()), float)
		test_pic.append(p)

test_pic = np.array(test_pic)
print test_pic.shape

result = []
for i in range(0, test_pic.shape[0]):
	t_pic_ld = test_pic[i] - averageface
	print t_pic_ld.shape
	print eigen.shape
	t_pic_ld = np.dot(t_pic_ld, eigen.T)
	r = clf.predict([t_pic_ld])
	result.append(r)

for i in range(0, len(result)):
	img = IMG.fromarray(test_pic[i].reshape(orgshape).astype(np.uint8))
	img.save("result/" + str(i) + "test.jpg")
	k = 0;
	for j in range(0, len(pic_sign)):
		if pic_sign[j] == result[i]:
			k = j
			break
	img = IMG.fromarray(pic_seto[k].reshape(orgshape).astype(np.uint8))
	img.save("result/" + str(i) + "match.jpg")