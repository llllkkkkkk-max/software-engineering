import pickle
from collections import Counter

def load_pickle(filename):
    """从pickle文件中加载数据"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def count_occurrences(arr, target):
    """计算目标元素在列表中出现的次数"""
    res = arr.count(target)
    return res

def separate_single_multiple(filepath, save_single_path, save_multiple_path):
    """将语料中的单候选和多候选分隔开"""
    with open(filepath, 'r') as f:
        total_data = eval(f.read())
    qids = []
    for i in range(len(total_data)):
        qids.append(total_data[i][0][0])
    result = Counter(qids)
    for i in range(len(total_data)):
        if result[total_data[i][0][0]] == 1:
            qids.append(total_data[i][0][0])

def convert_single_to_labeled(path1, path2):
    """将单候选数据转换为带有标签的格式"""
    total_data = load_pickle(path1)
    labels = [[data[0][0], 1] for data in total_data]
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    with open(path2, 'w') as f:
        f.write(str(total_data_sort))


if __name__ == "__main__":
    # 将staqc_python中的单候选和多候选分开
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = '../hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    separate_single_multiple(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    # 将staqc_sql中的单候选和多候选分开
    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = '../hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = '../hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    separate_single_multiple(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    # 将large_python中的单候选和多候选分开
    large_python_path = '../hnn_process/ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    separate_single_multiple(large_python_path, large_python_single_save, large_python_multiple_save)

    # 将large_sql中的单候选和多候选分开
    large_sql_path = '../hnn_process/ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    separate_single_multiple(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = '../hnn_process/ulabel_data/large_corpus/single/python_large_single_label.txt'
    convert_single_to_labeled(large_sql_single_save, large_sql_single_label_save)
    convert_single_to_labeled(large_python_single_save, large_python_single_label_save)
