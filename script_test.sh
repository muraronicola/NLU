cd ./248449_nicola_muraro/LM/part_1/
echo "LM part 1"

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt


cd ../part_2/
echo "LM part 2"
time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt






cd ../../NLU/part1/
echo "NLU part 1"
time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt


cd ../part_2/
echo "NLU part 2"
time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt








cd ../../SA/part1/
echo "SA part 1"
time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt
