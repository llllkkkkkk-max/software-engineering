import pickle
#构建初步词典的具体步骤1
def get_vocab(corpus1, corpus2):
    word_vocab = set()
    corpora = [corpus1, corpus2]
    for corpus in corpora:
        for data in corpus:
            for i in range(len(data[1][0])):
                word_vocab.add(data[1][0][i])
            for i in range(len(data[1][1])):
                word_vocab.add(data[1][1][i])
            for i in range(len(data[2][0])):
                word_vocab.add(data[2][0][i])
            for i in range(len(data[3])):
                word_vocab.add(data[3][i])
    print(len(word_vocab))
    return word_vocab



def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')
    return data



def get_vocab(corpus1, corpus2):
    word_vocab = set()
    corpora = [corpus1, corpus2]
    for corpus in corpora:
        for data in corpus:
            for sublist in data[1:3]:
                word_vocab.update(sublist2 for sublist2 in sublist)
            word_vocab.update(data[3])
    print(len(word_vocab))
    return word_vocab



def vocab_prpcessing(filepath1, filepath2, save_path):
    total_data1 = load_pickle(filepath1)
    total_data2 = load_pickle(filepath2)
    x1 = get_vocab(total_data1, total_data2)
    with open(save_path, "w") as f:
        f.write(str(x1))



def final_vocab_prpcessing(filepath1, filepath2, save_path):
    total_data1 = set(load_pickle(filepath1))
    total_data2 = load_pickle(filepath2)
    x1 = get_vocab(total_data1, total_data2)
    word_set = x1 - total_data1
    print(len(total_data1))
    print(len(word_set))
    with open(save_path, "w") as f:
        f.write(str(word_set))



if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    # vocab_prpcessing(python_hnn,python_staqc,python_word_dict)
    # vocab_prpcessing(sql_hnn,sql_staqc,sql_word_dict)
    #====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    final_vocab_prpcessing(sql_word_dict, new_sql_large, large_word_dict_sql)
    #vocab_prpcessing(new_sql_staqc,new_sql_large,final_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    #final_vocab_prpcessing(python_word_dict, new_python_large, large_word_dict_python)
    #vocab_prpcessing(new_python_staqc,new_python_large,final_word_dict_python)





