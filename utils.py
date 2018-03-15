from six.moves import cPickle

def path_reader(filename):
    with open(filename) as f:
        path_list = f.read().splitlines()
    return path_list

def load_dataset(file_path):
    with open(file_path, 'rb') as cPickle_file:
        [X_train, y_train, X_val, y_val, X_test, y_test] = cPickle.load(cPickle_file)
    return X_train, y_train, X_val, y_val, X_test, y_test