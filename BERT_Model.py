import torch
from pytorch_transformers import BertTokenizer, BertModel

class DropoutClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 200):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        #self.bn1 = torch.nn.BatchNorm1d(num_features=hidden_size)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout)
        nextout = nextout.clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear3(nextout)
        return nextout
    
    def skip_rows(self, i):
        if i % 14 == 12:
            return False
        else:
            return True
    
    def data_processer(self, t):
        return t

class Naive_BertForWordSegmentation(torch.nn.Module):
    def __init__(self):
        super(Naive_BertForWordSegmentation, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        processed_list = []
        assert(len(input_tokens) == hidden_states[0].shape[1])
        for i in range(len(indexed_tokens) - 1):
            t_1 = hidden_states[0][0][i]
            t_2 = hidden_states[0][0][i + 1]
            processed_list.append(torch.unsqueeze(torch.cat((t_1, t_2), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss

class BertForWordSegmentation(torch.nn.Module):
    def __init__(self):
        super(BertForWordSegmentation, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        processed_list = []
        assert(len(input_tokens) == hidden_states[0].shape[1])
        for i in range(len(indexed_tokens) - 1):
            t_1 = hidden_states[12][0][i]
            t_2 = hidden_states[12][0][i + 1]
            processed_list.append(torch.unsqueeze(torch.cat((t_1, t_2), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss