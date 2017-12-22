cp $1 test.csv
cp $3 movies.csv
cp $4 users.csv
cd model_dnn/
python3 test.py
cd ..
mv model_dnn/output.txt $2