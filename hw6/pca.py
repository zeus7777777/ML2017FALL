import skimage.io
import numpy as np
import os
import sys

images = []
for f in os.listdir(sys.argv[1]):
    images.append(skimage.io.imread(os.path.join(sys.argv[1], f)))
print('Images read\n')
images = np.array(images, dtype=np.float)

images = np.reshape(images, [-1, 600*600*3])
mean = np.mean(images, axis=0)
images -= mean

v = []

u, s, v = np.linalg.svd(images.T, full_matrices=False)

del images, v

e_vector = u.T
print(e_vector.shape)
print(e_vector)
print()

path = os.path.join(sys.argv[1], sys.argv[2])
target = skimage.io.imread(path).reshape([600*600*3]) - mean

rc = np.zeros([600*600*3])
ww = []
for i in range(4):
    w = np.inner(target, e_vector[i])
    ww.append(i)
    rc += w*e_vector[i]
for i in range(len(ww)):
    print(s[ww[i]]/sum(s))
rc += mean
rc = rc.reshape([600, 600, 3])
rc -= np.min(rc)
rc /= np.max(rc)
rc = (rc*255).astype(np.uint8)

skimage.io.imsave('reconstruction.jpg', rc)