from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gensim.models import KeyedVectors
import operator


    #将文本格式的Word2Vec模型转换为二进制格式并保存到指定路径
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)


    #生成新的词典和词向量
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    model = KeyedVectors.load(type_vec_path, mmap='r')
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']
    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)
    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)
    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)


    #将输入的文本转换为对应的索引列表
def get_index(type, text, word_dict):
    location = []
    if type == 'code':
        location.append(1)
        len_c = len(text)
        if len_c + 1 < 350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, len_c):
                    if word_dict.get(text[i]) is not None:
                        index = word_dict.get(text[i])
                        location.append(index)
                    else:
                        index = word_dict.get('UNK')
                        location.append(index)
                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) is not None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
    return location


    #对类型序列进行处理和序列化
def Serialization(word_dict_path, type_path, final_type_path):
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)
    with open(type_path, 'r') as f:
        text_list = f.readlines()
    type_list = []
    for text in text_list:
        text = text.strip().split('\t')
        location = get_index('type', text, word_dict)
        type_list.append(location)
    with open(final_type_path, 'wb') as f:
        pickle.dump(type_list, f)


    #词向量可视化
def Visualization(word_vectors_path, word_dict_path):
    with open(word_vectors_path, 'rb') as f:
        word_vectors = pickle.load(f)
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(word_vectors[:1000, :])
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(word_dict.keys(), Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()


    #扩充词典和词向量
def get_new_dict_append(type_vec_path, previous_dict, previous_vec, append_word_path, final_vec_path, final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961
    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)
    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)
    with open(append_word_path,'r')as f:
        append_word= eval(f.read())
        f.close()
    # 输出词向量
    print(type(pre_word_vec))
    word_dict = list(pre_word_dict.keys()) #'#其中0 PAD_ID,1SOS_ID,2E0S_ID,3UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word = []
    print(len(append_word))
    rng = np.random.RandomState(None)
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    for word in append_word:
        try:
            word_vectors.append(model.wv[word]) #加载词向量
            word_dict.append(word)
        except:
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])
    #判断词向量是否正确
    print("----------------------------")
    couunt = 0
    for i in range(159035,len(word_dict)):
        if operator.eq(word_vectors[i].tolist(), model.wv[word_dict[i]].tolist()) == True:
            continue
        else:
            couunt +=1
    print(couunt)
    word_vectors = np.array(word_vectors)
    print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)
    print("完成")


if __name__ == '__main__':
    # define file paths
    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt'  # 239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'  # 2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # preprocess data and create word dict
    preprocess_and_create_word_dict(ps_path, ps_path_bin, sql_path, sql_path_bin, python_word_path,
                                    python_word_vec_path, python_word_dict_path, sql_word_path, sql_word_vec_path,
                                    sql_word_dict_path)

    # create label data
    create_label_data(new_sql_staqc, sql_final_word_dict_path, new_sql_large, "large_sql_f", new_python_staqc,
                      python_final_word_dict_path, new_python_large, "large_python_f", "staqc_sql_f", "staqc_python_f")

    print('序列化完毕')