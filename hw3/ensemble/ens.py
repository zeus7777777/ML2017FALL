import csv
import numpy as np

with open('13.txt', 'r') as f:
        reader = csv.reader(f)
        i=0
        pred1 = []
        for row in reader:
                i += 1
                if i==1:
                        continue
                pred1.append(int(row[1]))
print(pred1)

with open('16.txt', 'r') as f:
        reader = csv.reader(f)
        i=0
        pred2 = []
        for row in reader:
                i += 1
                if i==1:
                        continue
                pred2.append(int(row[1]))
print(pred2)

with open('19.txt', 'r') as f:
        reader = csv.reader(f)
        i=0
        pred3 = []
        for row in reader:
                i += 1
                if i==1:
                        continue
                pred3.append(int(row[1]))
print(pred3)

with open('11.txt', 'r') as f:
        reader = csv.reader(f)
        i=0
        pred4 = []
        for row in reader:
                i += 1
                if i==1:
                        continue
                pred4.append(int(row[1]))
print(pred4)



vote = np.zeros([len(pred1), 7])
for i in range(len(pred1)):
	vote[i][pred1[i]] += 1
	vote[i][pred2[i]] += 1
	vote[i][pred3[i]] += 1
	vote[i][pred4[i]] += 1
ans = np.argmax(vote, axis=-1)
print(ans)

with open('ans.txt', 'w') as f:
	f.write('id,label\n')
	for i in range(len(pred1)):
		f.write(str(i)+','+str(ans[i])+'\n')
		
