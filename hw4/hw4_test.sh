cp $1 testing_data.txt
cd model7/
python3 test_rnn.py
cp output.txt ../$2
