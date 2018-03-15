import settings
from utils import *

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(settings.TARGET_PATH)
    print(X_train[0])
    print(y_train[0])