import numpy as np
import glob
from scipy import misc
data_dir = '../../data/raw/Ara2013-Canon/'
target_dir = '../../data/split/'
def get_data():

    x_train = np.array([np.array(misc.imread(fname), dtype=np.int32) for fname in sorted(glob.glob(data_dir+'*rgb.png'))])
    y_train = [np.array(misc.imread(fname), dtype=np.int32) for fname in sorted(glob.glob(data_dir+'*label.png'))]

    x_resize = []
    for img in x_train:
        img = misc.imresize(img,(128,128,3))
        x_resize.append(np.array([img]))

    x_resize = np.array(x_resize)

    y_seqs = []



    max_sum = 0

    print(y_train[0].shape)

    for temp_img in y_train:
        colors = set(tuple(v) for m2d in temp_img for v in m2d)

        sums = [sum(color) for color in colors]
        sums.remove(0)

        sequence = np.zeros((13, 128,128,1))
        if(len(sums) > max_sum):
            max_sum = len(sums)
        temp_sums = np.sum(temp_img, axis=2)
        for j,i in enumerate(sums):
            inds = temp_sums == i
            temp = np.zeros(temp_img.shape[:2])
            temp[inds] = 1
            temp = misc.imresize(temp, (128,128))
            inds = temp > 0
            temp[inds] = 1

            temp = temp.reshape((128,128,1))
            sequence[j,:,:] = temp


        y_seqs.append(sequence)


    print "Max Sum:",max_sum
    return x_resize,y_seqs


x_train,y_train = get_data()

x_train = np.array(x_train)
y_train = np.array(y_train)

np.save('../../data/split/x_train',x_train)
np.save('../../data/split/y_train',y_train)