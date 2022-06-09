from pytorch_transformers import BertTokenizer
import nltk.data

def is_english(character):
    if 'A' <= character and character <= 'Z':
        return True
    if 'Ａ' <= character and character <= 'Ｚ':
        return True
    if 'a' <= character and character <= 'z':
        return True
    if 'ａ' <= character and character <= 'ｚ':
        return True
    if '0' <= character and character <= '9':
        return True
    if '０' <= character and character <= '９':
        return True
    if character == '○':
        return True
    if character == '.' or character == '．' or character == '%' or character == '％':
        return True
    return False

def read_from_training_data(filename):
    characters = open(filename).read()
    #x_list is a list of sentences, which means x_list is a 2-d array
    sentence_cnt = 0
    x_list = [[]]
    y_list = [[]]
    i = 0
    english_before = False
    for i in range(len(characters)):
        if is_english(characters[i]):
            if english_before:
                continue
            else:
                english_before = True
                continue
        else:
            if english_before:
                x_list[sentence_cnt].append('#')
                y_list[sentence_cnt].append(characters[i] == ' ')
                english_before = False
                
        if not characters[i] == ' ':
            if characters[i] == '\n':
                if not len(x_list[sentence_cnt]) == 0:
                    sentence_cnt += 1
                    x_list.append([])
                    y_list.append([])
                continue
        
            y_list[sentence_cnt].append((characters[i + 1] == ' '))
            x_list[sentence_cnt].append(characters[i])
            
            if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
                
    i = 0
    l = len(x_list)
    while i < l:
        if(len(x_list[i])) > 512:
            del x_list[i]
            del y_list[i]
            i -= 1
            l -= 1
        i += 1
            
    if len(x_list[len(x_list) - 1]) == 0:
        del x_list[len(x_list) - 1]
        del y_list[len(y_list) - 1]
    
    return x_list, y_list

def read_from_testing_data(filename):
    characters = open(filename).read()
    sentence_cnt = 0
    x_list = [[]]
    english_before = False
    for i in range(len(characters)):
        if is_english(characters[i]):
            if english_before:
                continue
            else:
                english_before = True
                continue
        else:
            if english_before:
                x_list[sentence_cnt].append('#')
                english_before = False
        if characters[i] == '\n':
            if not len(x_list[sentence_cnt]) == 0:
                sentence_cnt += 1
                x_list.append([])
            continue
        x_list[sentence_cnt].append(characters[i])
        
        if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
            sentence_cnt += 1
            x_list.append([])
        
    return [x for x in x_list if x != []]

def sentenceReader(filename, file_type):
    assert(file_type == 'testing' or file_type == 'training')
    if(file_type == 'testing'):
        return read_from_testing_data(filename)
    if(file_type == 'training'):
        x_list, _ = read_from_training_data(filename)
        return x_list