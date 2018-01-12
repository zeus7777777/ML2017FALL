cp $1 image.npy
python3 cluster.py
cp $2 test_case.csv
python3 test.py
mv output.txt $3