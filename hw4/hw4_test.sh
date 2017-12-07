cd model7/
cp ../$1 testing_data.txt
python3 test_rnn.py
cp output.txt ../$2