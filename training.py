import torch
import random
from BERT_Model import BertForWordSegmentation, Naive_BertForWordSegmentation
from GNN import Naive_GCN, GCN, Naive_GCN_lexicon, GCN_lexicon
from ReadData import is_english, read_from_training_data, read_from_testing_data
    
def data_loader(x_list, y_list):
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)

    for i in range(len(x_tuple)):
        yield x_tuple[i], y_tuple[i]

def prepare_xy_list(filename, num_sentences, eliminate_one = True):
    x_list, y_list = read_from_training_data(filename)
    print("There are %d sentences in this file." % (len(x_list)))
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)
    x_list = list(x_tuple)
    y_list = list(y_tuple)
    
    cnt = 0
    l = len(x_list)
    while cnt < l:
        if len(x_list[cnt]) == 0:
            del x_list[cnt]
            del y_list[cnt]
            l -= 1
            continue
        if eliminate_one and len(x_list[cnt]) == 1:
            del x_list[cnt]
            del y_list[cnt]
            l -= 1
            continue
        cnt += 1
    
    return x_list[:num_sentences], y_list[:num_sentences]

def train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one):
    x_list, y_list = prepare_xy_list(train_file, num_sentences, eliminate_one)
    
    partition = int(len(x_list) * 4 / 5)

    best_acc = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        train_loader = data_loader(x_list[:partition], y_list[:partition])
        test_loader = data_loader(x_list[partition:], y_list[partition:])
        batch_cnt = 0
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            batch_cnt += 1
        print ('Epoch: [%d/%d], Average Loss: %.4f' % (epoch+1, num_epochs, total_loss / batch_cnt))
            
        num_characters = 0
        correct_predictions = 0
        model.eval()
        for x, y in test_loader:
            z, _ = model(x)
            for (i, output) in enumerate(z):
                if y[i] == output.argmax():
                    correct_predictions += 1
                num_characters += 1
        print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
        if correct_predictions * 1.0 / num_characters > best_acc:
            best_acc = correct_predictions * 1.0 / num_characters
            if do_save:
                torch.save(model, save_path)
        torch.cuda.empty_cache()

def test_model(model_path, filename, num_sentences):
    model = torch.load(model_path)
    model.eval()
    
    x_list, y_list = prepare_xy_list(filename, num_sentences)
    
    num_characters = 0
    correct_predictions = 0
    test_loader = data_loader(x_list[:num_sentences], y_list[:num_sentences])
    for x, y in test_loader:
        z, _ = model(x)
        for (i, output) in enumerate(z):
            if y[i] == output.argmax():
                correct_predictions += 1
            num_characters += 1
    print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
    
        
def train_Naive_BERT(train_file, num_sentences, num_epochs = 20, learning_rate = 0.005, do_save = True, save_path = 'Naive_BERT.bin', eliminate_one = True):
    model = Naive_BertForWordSegmentation()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()

        
def train_BERT(train_file, num_sentences, num_epochs = 20, learning_rate = 0.005, do_save = True, save_path = 'BERT.bin', eliminate_one = True):
    model = BertForWordSegmentation()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()

def train_Naive_GCN(train_file, num_sentences, num_epochs = 30, learning_rate = 0.001, do_save = True, save_path = 'Naive_GCN.bin', eliminate_one = True):
    model = Naive_GCN()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()

def train_GCN(train_file, num_sentences, num_epochs = 30, learning_rate = 0.001, do_save = True, save_path = 'GCN.bin', eliminate_one = True):
    model = GCN()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()

def train_Naive_GCN_lexicon(train_file, num_sentences, num_epochs = 30, learning_rate = 0.001, do_save = True, save_path = 'Naive_GCN_lexicon.bin', eliminate_one = True):
    model = Naive_GCN_lexicon()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()

def train_GCN_lexicon(train_file, num_sentences, num_epochs = 30, learning_rate = 0.001, do_save = True, save_path = 'GCN_lexicon.bin', eliminate_one = True):
    model = GCN_lexicon()
    train(train_file, num_sentences, model, num_epochs, learning_rate, do_save, save_path, eliminate_one)
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    torch.cuda.empty_cache()
    
    print("GCN_lexicon:")
    train_GCN_lexicon("pku_training.txt", 2000, 30, learning_rate = 0.001)
    test_model("GCN_lexicon.bin", "pku_test_gold.txt", 5000)
    print("\n")
    
    print("GCN:")
    train_GCN("pku_training.txt", 2000, 30, learning_rate = 0.001)
    test_model("GCN.bin", "pku_test_gold.txt", 5000)
    print("\n")
    
    print("Naive_GCN:")
    train_Naive_GCN("pku_training.txt", 2000, 30, learning_rate = 0.001)
    test_model("Naive_GCN.bin", "pku_test_gold.txt", 5000)
    print("\n")