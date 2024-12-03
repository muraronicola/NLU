cd ./248449_nicola_muraro/LM/part_1/

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt


cd ../part_2/

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt






cd ../../NLU/part1/

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt


cd ../part_2/

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt








cd ../../SA/part1/

time python3 ./main.py -e=True --model=./bin/best_model.pt
time python3 ./main.py -s=True 
time python3 ./main.py -e=True --model=./bin/best_model_0.pt
