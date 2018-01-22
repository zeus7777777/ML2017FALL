cp $1 src/data/test.data
cp $2 src/data/test.csv

cd src/

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
cd ../
cp new1/output.txt ensemble/1.txt

python3 ensemble.py
cd ../

mv src/ans.txt $3
