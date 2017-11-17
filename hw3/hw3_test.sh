cp $1 ensemble/$1
cd ensemble/
wget http://cmlab.csie.ntu.edu.tw/~zeus/model.tar
tar xcf model.tar
python3 vgg11.py $1 11.txt
python3 vgg13.py $1 13.txt
python3 vgg16.py $1 16.txt
python3 vgg19.py $1 19.txt
python3 ens.py
cd ..
mv ensemble/ans.txt $2