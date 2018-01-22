import csv
import numpy as np
import os

files = os.listdir('ensemble')
total = len(files)

pred = []
for i in range(total):
    with open(os.path.join('ensemble' ,files[i]), 'r') as f:
        reader = csv.reader(f)
        i=0
        p = []
        for row in reader:
            i += 1
            if i==1: continue
            p.append(int(row[1]))
    pred.append(p[:])

vote = np.zeros([len(pred[0]), 4])
for i in range(len(pred[0])):
    for j in range(total):
        vote[i][pred[j][i]] += 1
ans = np.argmax(vote, axis=-1)
print(vote)

with open('ans.txt', 'w') as f:
	f.write('id,answer\n')
	for i in range(len(ans)):
		f.write(str(i+1)+','+str(ans[i])+'\n')