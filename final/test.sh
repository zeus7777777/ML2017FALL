cp $1 data/test.data
cp $2 data/test.csv

cd model2/
python3 preprocess.py
python3 test.py
cd ../
cp model2/output.txt ensemble/2.txt

cd new0/
python3 preprocess.py
python3 test.py
cd ../
cp new0/output.txt ensemble/0.txt

cd new1/
python3 preprocess.py
python3 test.py
cd ..
cp new1/output.txt ensemble/1.txt

python3 ensemble.py
mv ans.txt $3
