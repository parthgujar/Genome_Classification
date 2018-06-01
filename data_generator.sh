rm -rf Train_1
mkdir Train_1
python sticky_snippet_generator.py 2500 0 1 out1.txt
python sticky_snippet_generator.py 2500 0 2 out2.txt
cat out1.txt out2.txt > ./Train_1/file1.txt
python sticky_snippet_generator.py 2500 0 3 out1.txt
python sticky_snippet_generator.py 2500 0 4 out2.txt
cat out1.txt out2.txt > ./Train_1/file2.txt
python sticky_snippet_generator.py 2500 0 5 out1.txt
python sticky_snippet_generator.py 2500 0 6 out2.txt
cat out1.txt out2.txt > ./Train_1/file3.txt
python sticky_snippet_generator.py 2500 0 7 out1.txt
python sticky_snippet_generator.py 2500 0 8 out2.txt
cat out1.txt out2.txt > ./Train_1/file4.txt
python sticky_snippet_generator.py 2500 0 0 out1.txt
python sticky_snippet_generator.py 2500 0 0 out2.txt
cat out1.txt out2.txt > ./Train_1/file5.txt
python sticky_snippet_generator.py 2500 0 20 out1.txt
python sticky_snippet_generator.py 2500 0 20 out2.txt
cat out1.txt out2.txt > ./Train_1/file6.txt
rm out1.txt
rm out2.txt

rm -rf Train_2
mkdir Train_2
python sticky_snippet_generator.py 5000 0 1 out1.txt
python sticky_snippet_generator.py 5000 0 2 out2.txt
cat out1.txt out2.txt > ./Train_2/file1.txt
python sticky_snippet_generator.py 5000 0 3 out1.txt
python sticky_snippet_generator.py 5000 0 4 out2.txt
cat out1.txt out2.txt > ./Train_2/file2.txt
python sticky_snippet_generator.py 5000 0 5 out1.txt
python sticky_snippet_generator.py 5000 0 6 out2.txt
cat out1.txt out2.txt > ./Train_2/file3.txt
python sticky_snippet_generator.py 5000 0 7 out1.txt
python sticky_snippet_generator.py 5000 0 8 out2.txt
cat out1.txt out2.txt > ./Train_2/file4.txt
python sticky_snippet_generator.py 5000 0 0 out1.txt
python sticky_snippet_generator.py 5000 0 0 out2.txt
cat out1.txt out2.txt > ./Train_2/file5.txt
python sticky_snippet_generator.py 5000 0 20 out1.txt
python sticky_snippet_generator.py 5000 0 20 out2.txt
cat out1.txt out2.txt > ./Train_2/file6.txt
rm out1.txt
rm out2.txt


rm -rf Train_3
mkdir Train_3
python sticky_snippet_generator.py 10000 0 1 out1.txt
python sticky_snippet_generator.py 10000 0 2 out2.txt
cat out1.txt out2.txt > ./Train_3/file1.txt
python sticky_snippet_generator.py 10000 0 3 out1.txt
python sticky_snippet_generator.py 10000 0 4 out2.txt
cat out1.txt out2.txt > ./Train_3/file2.txt
python sticky_snippet_generator.py 10000 0 5 out1.txt
python sticky_snippet_generator.py 10000 0 6 out2.txt
cat out1.txt out2.txt > ./Train_3/file3.txt
python sticky_snippet_generator.py 10000 0 7 out1.txt
python sticky_snippet_generator.py 10000 0 8 out2.txt
cat out1.txt out2.txt > ./Train_3/file4.txt
python sticky_snippet_generator.py 10000 0 0 out1.txt
python sticky_snippet_generator.py 10000 0 0 out2.txt
cat out1.txt out2.txt > ./Train_3/file5.txt
python sticky_snippet_generator.py 10000 0 20 out1.txt
python sticky_snippet_generator.py 10000 0 20 out2.txt
cat out1.txt out2.txt > ./Train_3/file6.txt
rm out1.txt
rm out2.txt

rm -rf Train_4
mkdir Train_4
python sticky_snippet_generator.py 60000 0 20 out.txt
mv out.txt  ./Train_4/file.txt

echo "Train Data Generated"


rm -rf Test_1
mkdir Test_1
python sticky_snippet_generator.py 2500 0.2 1 out1.txt
python sticky_snippet_generator.py 2500 0.2 2 out2.txt
cat out1.txt out2.txt > ./Test_1/file1.txt
python sticky_snippet_generator.py 2500 0.2 3 out1.txt
python sticky_snippet_generator.py 2500 0.2 4 out2.txt
cat out1.txt out2.txt > ./Test_1/file2.txt
python sticky_snippet_generator.py 2500 0.2 5 out1.txt
python sticky_snippet_generator.py 2500 0.2 6 out2.txt
cat out1.txt out2.txt > ./Test_1/file3.txt
python sticky_snippet_generator.py 2500 0.2 7 out1.txt
python sticky_snippet_generator.py 2500 0.2 8 out2.txt
cat out1.txt out2.txt > ./Test_1/file4.txt
python sticky_snippet_generator.py 2500 0.2 0 out1.txt
python sticky_snippet_generator.py 2500 0.2 0 out2.txt
cat out1.txt out2.txt > ./Test_1/file5.txt
python sticky_snippet_generator.py 2500 0.2 20 out1.txt
python sticky_snippet_generator.py 2500 0.2 20 out2.txt
cat out1.txt out2.txt > ./Test_1/file6.txt
rm out1.txt
rm out2.txt

rm -rf Test_2
mkdir Test_2
python sticky_snippet_generator.py 2500 0.4 1 out1.txt
python sticky_snippet_generator.py 2500 0.4 2 out2.txt
cat out1.txt out2.txt > ./Test_2/file1.txt
python sticky_snippet_generator.py 2500 0.4 3 out1.txt
python sticky_snippet_generator.py 2500 0.4 4 out2.txt
cat out1.txt out2.txt > ./Test_2/file2.txt
python sticky_snippet_generator.py 2500 0.4 5 out1.txt
python sticky_snippet_generator.py 2500 0.4 6 out2.txt
cat out1.txt out2.txt > ./Test_2/file3.txt
python sticky_snippet_generator.py 2500 0.4 7 out1.txt
python sticky_snippet_generator.py 2500 0.4 8 out2.txt
cat out1.txt out2.txt > ./Test_2/file4.txt
python sticky_snippet_generator.py 2500 0.4 0 out1.txt
python sticky_snippet_generator.py 2500 0.4 0 out2.txt
cat out1.txt out2.txt > ./Test_2/file5.txt
python sticky_snippet_generator.py 2500 0.4 20 out1.txt
python sticky_snippet_generator.py 2500 0.4 20 out2.txt
cat out1.txt out2.txt > ./Test_2/file6.txt
rm out1.txt
rm out2.txt





rm -rf Test_3
mkdir Test_3
python sticky_snippet_generator.py 2500 0.6 1 out1.txt
python sticky_snippet_generator.py 2500 0.6 2 out2.txt
cat out1.txt out2.txt > ./Test_3/file1.txt
python sticky_snippet_generator.py 2500 0.6 3 out1.txt
python sticky_snippet_generator.py 2500 0.6 4 out2.txt
cat out1.txt out2.txt > ./Test_3/file2.txt
python sticky_snippet_generator.py 2500 0.6 5 out1.txt
python sticky_snippet_generator.py 2500 0.6 6 out2.txt
cat out1.txt out2.txt > ./Test_3/file3.txt
python sticky_snippet_generator.py 2500 0.6 7 out1.txt
python sticky_snippet_generator.py 2500 0.6 8 out2.txt
cat out1.txt out2.txt > ./Test_3/file4.txt
python sticky_snippet_generator.py 2500 0.6 0 out1.txt
python sticky_snippet_generator.py 2500 0.6 0 out2.txt
cat out1.txt out2.txt > ./Test_3/file5.txt
python sticky_snippet_generator.py 2500 0.6 20 out1.txt
python sticky_snippet_generator.py 2500 0.6 20 out2.txt
cat out1.txt out2.txt > ./Test_3/file6.txt
rm out1.txt
rm out2.txt



echo "Test Data Generated" 
