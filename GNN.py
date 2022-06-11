import torch
from pytorch_transformers import BertTokenizer, BertModel
from BERT_Model import DropoutClassifier
from torch_geometric.nn import GCNConv
import jieba

class Naive_GCN(torch.nn.Module):
    def __init__(self):
        super(Naive_GCN, self).__init__()
        self.conv1 = GCNConv(768, 768).to('cuda')
        self.conv2 = GCNConv(768, 768).to('cuda')
        self.conv3 = GCNConv(768, 768).to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        assert(len(input_tokens) == hidden_states[0].shape[1])
        
        edge_index_list = []
        for i in range(len(input_tokens) - 1):
            edge_index_list.append([i, i + 1])
            edge_index_list.append([i + 1, i])
            
        edge_index = torch.LongTensor(edge_index_list).t().contiguous().to('cuda')
        
        x = hidden_states[0][0].to('cuda')
        h = self.conv1(x, edge_index)
        h = torch.nn.functional.relu(h)
        h = self.conv2(h, edge_index)
        h = torch.nn.functional.relu(h)
        h = self.conv3(h, edge_index)
        h = torch.nn.functional.relu(h)
        
        assert(len(h) == len(input_tokens))
        processed_list = []
        for i in range(len(h) - 1):
            processed_list.append(torch.unsqueeze(torch.cat((h[i], h[i + 1]), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss
    
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(768, 768).to('cuda')
        self.conv2 = GCNConv(768, 768).to('cuda')
        self.conv3 = GCNConv(768, 768).to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        assert(len(input_tokens) == hidden_states[0].shape[1])
        
        edge_index_list = []
        for i in range(len(input_tokens) - 1):
            edge_index_list.append([i, i + 1])
            edge_index_list.append([i + 1, i])
            
        edge_index = torch.LongTensor(edge_index_list).t().contiguous().to('cuda')
        
        x = hidden_states[12][0].to('cuda')
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        
        assert(len(h) == len(input_tokens))
        processed_list = []
        for i in range(len(h) - 1):
            processed_list.append(torch.unsqueeze(torch.cat((h[i], h[i + 1]), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss

class Naive_GCN_lexicon(torch.nn.Module):
    def __init__(self):
        super(Naive_GCN_lexicon, self).__init__()
        self.conv1 = GCNConv(768, 768).to('cuda')
        self.conv2 = GCNConv(768, 768).to('cuda')
        self.conv3 = GCNConv(768, 768).to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        assert(len(input_tokens) == hidden_states[0].shape[1])
        
        sentence = ''.join(input_tokens)
        
        l_cut = jieba.lcut(sentence, cut_all = True)
        l_cut_np = []
        [l_cut_np.append(x) for x in l_cut if x not in l_cut_np]
        
        edge_index_list = []
        for w in l_cut_np:
            if len(w) < 2:
                continue
            index = sentence.find(w)
            while index != -1:
                edge_index_list.append([index, index + len(w) - 1])
                edge_index_list.append([index + len(w) - 1, index])
                index = sentence.find(w, index + 1)
            
        edge_index = torch.LongTensor(edge_index_list).t().contiguous().to('cuda')
        
        x = hidden_states[0][0].to('cuda')
        
        # If there are no edges, then there's no need to do GCN.
        if edge_index.shape[0] == 0:
            h = x
        else:
            h = self.conv1(x, edge_index)
            h = h.tanh()
            h = self.conv2(h, edge_index)
            h = h.tanh()
            h = self.conv3(h, edge_index)
            h = h.tanh()
            
        assert(len(h) == len(input_tokens))
        processed_list = []
        for i in range(len(h) - 1):
            processed_list.append(torch.unsqueeze(torch.cat((h[i], h[i + 1]), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss

class GCN_lexicon(torch.nn.Module):
    def __init__(self):
        super(GCN_lexicon, self).__init__()
        self.conv1 = GCNConv(768, 768).to('cuda')
        self.conv2 = GCNConv(768, 768).to('cuda')
        self.conv3 = GCNConv(768, 768).to('cuda')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        hidden_states = outputs[2]
        assert(len(input_tokens) == hidden_states[0].shape[1])
        
        sentence = ''.join(input_tokens)
        
        l_cut = jieba.lcut(sentence, cut_all = True)
        l_cut_np = []
        [l_cut_np.append(x) for x in l_cut if x not in l_cut_np]
        
        edge_index_list = []
        for w in l_cut_np:
            if len(w) < 2:
                continue
            index = sentence.find(w)
            while index != -1:
                edge_index_list.append([index, index + len(w) - 1])
                edge_index_list.append([index + len(w) - 1, index])
                index = sentence.find(w, index + 1)
            
        edge_index = torch.LongTensor(edge_index_list).t().contiguous().to('cuda')
        
        x = hidden_states[12][0].to('cuda')
        
        # If there are no edges, then there's no need to do GCN.
        if edge_index.shape[0] == 0:
            h = x
        else:
            h = self.conv1(x, edge_index)
            h = h.tanh()
            h = self.conv2(h, edge_index)
            h = h.tanh()
            h = self.conv3(h, edge_index)
            h = h.tanh()
            
        assert(len(h) == len(input_tokens))
        processed_list = []
        for i in range(len(h) - 1):
            processed_list.append(torch.unsqueeze(torch.cat((h[i], h[i + 1]), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss