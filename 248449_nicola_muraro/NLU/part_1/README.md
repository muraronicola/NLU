For executing the code, you can call it without any arguments.
In this case, the code will train a model for each point of the assignment, and it will evaluate it.


Still, there are some optional arguments that you can use to customize the execution of the code.
The optional arguments are:
- -d, --device: The device to be used for the various experiments (default: 'cuda:0')
- -s, --save: Save the best model (obtained during the training) to disk (default: False)
- -e, --eval_only: Evaluate only the model saved on the disk, without training any new model (default: False)
- -m, --model: The model path to be evaluated (used only with eval_only) (default: ./bin/best_model.pt)