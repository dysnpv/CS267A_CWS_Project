# CS267A_CWS_Project

"pku_training.txt" and "pku_test_gold.txt" are the datasets.

You need to install the pytorch geometric and jieba libraries to run the code.

Use the train_XXX functions in "training.py" to train models. Test them by using the test_model function in "training.py".

Navei_XXX means the model uses the first layer embedding of BERT. For instance, Naive_BERT is the model without the GCN layer and use the first layer embedding. If the model doesn't contain Naive, then it uses the last layer embedding.

(Naive_)BertForWordSegmentation are the BERT models. (Naive_)GCN are the GCN_S models (the GNN models with straightforward graph encoding). (Naive_)GCN_lexicon are the GCN_L models. 

All of the default parameters of the train_XXX functions are set to be identical to what we used in this project. Thus, you don't need to provide addtional arguments, just the training file (which is "pku_traning.txt") and the number of sentences you want the model to train on.

Specify the models you want to train and test in the main function in training.py, then run
	python training.py
or
	python3 training.py
to run the code.

we used cuda to speed up the traning process. If your GPU doesn't support it, well, I don't know what will happen. Turn cuda off if you meet any problems.

The training_logs folder contains all the training logs.

Our program writes the trained models to the disk. Make sure you have enough space left. Each model taks about 700mb space.
