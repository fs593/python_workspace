from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
import numpy as np
from PIL import Image as IMG
from compiler.ast import flatten
import os.path
import copy
#each person have numbers of face pic, using PCA to reduce the dimention of pic set
#in feature sub space, we use SVM to build a classifier of each person's face in sub feature space

pic_set = []
pic_sign = []
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
			pic_set.append(p)# add pic to the training set
			pic_sign.append(i)# labeling the data by using their id

pic_set = np.array(pic_set)
pic_seto = copy.deepcopy(pic_set)
print pic_set.shape

ncomponent = 80
pca = PCA(n_components = ncomponent).fit(pic_set)
pic_set_ld = pca.transform(pic_set)

param_grid = {'C': [0.1,0.5,1],  
              'gamma': [0.0005,0.00055,0.0006], }  
clf = GridSearchCV(svm.SVC(kernel = 'rbf'),param_grid)
clf.fit(pic_set_ld, pic_sign)
print "Best estimator found by grid search:"  
print clf.best_estimator_
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
t_pic_ld = pca.transform(test_pic)
for i in range(0, test_pic.shape[0]):
	r = clf.predict([t_pic_ld[i]])
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