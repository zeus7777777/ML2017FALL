mv $1 test.csv
cd model_mf/
python3 test.py
cd ..
cp model_mf/output.txt $2