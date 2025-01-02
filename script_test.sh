cd ./248449_nicola_muraro/LM/part_1/
echo "\n\nLM part 1"

echo "\nFirst execution"
time python3 ./main.py -e=True --model=./bin/best_model.pt
echo "\nSecond execution"
time python3 ./main.py -s=True 
echo "\nThird execution"
time python3 ./main.py -e=True --model=./bin/best_model_0.pt



cd ../part_2/
echo "\n\nLM part 2"

echo "\nFirst execution"
time python3 ./main.py -e=True --model=./bin/best_model.pt
echo "\nSecond execution"
time python3 ./main.py -s=True 
echo "\nThird execution"
time python3 ./main.py -e=True --model=./bin/best_model_0.pt






cd ../../
cd ./NLU/part_1/
echo "\n\nNLU part 1"

echo "\nFirst execution"
time python3 ./main.py -e=True --model=./bin/best_model.pt
echo "\nSecond execution"
time python3 ./main.py -s=True 
echo "\nThird execution"
time python3 ./main.py -e=True --model=./bin/best_model_0.pt



cd ../part_2/
echo "\n\nNLU part 2"

echo "\nFirst execution"
time python3 ./main.py -e=True --model=./bin/best_model.pt
echo "\nSecond execution"
time python3 ./main.py -s=True 
echo "\nThird execution"
time python3 ./main.py -e=True --model=./bin/best_model_0.pt








cd ../../
cd ./SA/part_1/
echo "\n\nSA part 1"

echo "\nFirst execution"
time python3 ./main.py -e=True --model=./bin/best_model.pt
echo "\nSecond execution"
time python3 ./main.py -s=True 
echo "\nThird execution"
time python3 ./main.py -e=True --model=./bin/best_model_0.pt
