mv $1 test.csv
mv $3 movies.csv
mv $4 users.csv
cd model_dnn/
python3 test.py
cd ..
mv model_dnn/output.txt $2