import sys
from PIL import Image

im = Image.open(sys.argv[1])
px = im.load()
x, y = im.size
#print(x,y)
for i in range(x):
    for j in range(y):
        px[i, j] = (int(px[i, j][0]/2), int(px[i, j][1]/2), int(px[i, j][2]/2))
im.save('Q2.png')
