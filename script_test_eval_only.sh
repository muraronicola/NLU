cd ./248449_nicola_muraro/LM/part_1/
echo "\n\nLM part 1"

time python3 ./main.py -e=True --model=./bin/best_model.pt 



cd ../part_2/
echo "\n\nLM part 2"

time python3 ./main.py -e=True --model=./bin/script_1_part_2_3_2__600_1.5_64_0.05_0.05_5.pt 




cd ../../
cd ./NLU/part_1/
echo "\n\nNLU part 1"

time python3 ./main.py -e=True --model=./bin/best_model.pt 



cd ../part_2/
echo "\n\nNLU part 2"

time python3 ./main.py -e=True --model=./bin/best_model.pt 







cd ../../
cd ./SA/part_1/
echo "\n\nSA part 1"
time python3 ./main.py -e=True --model=./bin/best_model.pt 
