import os.path
from PIL import Image as IMG

dir = "male/"
dir_list = []
for dirs in os.listdir(dir):
	dir_list.append(dirs)

for i in range(0, len(dir_list)):

	for rootdir, dirs, filenames in os.walk(dir + dir_list[i]):
		for j in range(0,3):
			l = len(filenames) - 1
			img = IMG.open(dir + dir_list[i] + "/" + filenames[l - j])
			img.save("test_sample/" + dir_list[i] + str(j) + ".jpg")
