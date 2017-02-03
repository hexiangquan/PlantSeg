import numpy as np
import glob
from scipy import misc

data_dir = '../../data/raw/Ara2013-Canon/'
target_dir = '../../data/processed/'

x_names = glob.glob(data_dir+'*rgb.png')
y_names = glob.glob(data_dir+'*label.png')
print(x_names)

x_train = np.array([np.array(misc.imresize(misc.imread(fname),(128,128)), dtype=np.int32) for fname in x_names])
y_train = np.array([np.array(misc.imresize(misc.imread(fname),(128,128)), dtype=np.int32) for fname in y_names])

for img,name in zip(x_train, x_names):
    misc.imresize(img, (128,128))
    name_string = name.split('/')
    misc.imsave(target_dir+name_string[len(name_string)-1],img)

for img,name in zip(y_train, y_names):
    misc.imresize(img, (128,128))
    name_string = name.split('/')
    print(name_string[len(name_string)-1])
    misc.imsave(target_dir+name_string[len(name_string)-1],img)