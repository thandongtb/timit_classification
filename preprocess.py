import os
import timeit; program_start_time = timeit.default_timer()
import random; random.seed(int(timeit.default_timer()))
from six.moves import cPickle
import numpy as np
import librosa
import python_speech_features as features
import settings
from utils import *

# Phonemes

phonemes = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh",
            "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y",
            "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
            "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

def get_total_duration(file):
    for line in reversed(list(open(file))):
        [_, val, _] = line.split()
        return int(val)

def find_phoneme (phoneme_idx):
    for i in range(len(phonemes)):
        if phoneme_idx == phonemes[i]:
            return i
    print("PHONEME NOT FOUND, NaN CREATED!")
    print("\t" + phoneme_idx + " wasn't found!")
    return -1

def create_mfcc(filename):
    sample, rate = librosa.load(filename)
    mfcc = features.mfcc(sample, rate, winlen=0.025, winstep=0.01, numcep = 13, nfilt=26,
                         preemph=0.97, appendEnergy=True)
    derivative = np.zeros(mfcc.shape)
    for i in range(1, mfcc.shape[0]-1):
        derivative[i, :] = mfcc[i+1, :] - mfcc[i-1, :]
    out = np.concatenate((mfcc, derivative), axis=1)

    return out, out.shape[0]

def calc_norm_param(X):
    total_len = 0
    mean_val = np.zeros(X[0].shape[1])
    std_val = np.zeros(X[0].shape[1])
    for obs in X:
        print(obs.shape)
        obs_len = obs.shape[0]
        mean_val += np.mean(obs, axis=0) * obs_len
        std_val += np.std(obs, axis=0) * obs_len
        total_len += obs_len
    mean_val /= total_len
    std_val /= total_len

    return mean_val, std_val, total_len

def normalize(X, mean_val, std_val):
    for i in range(len(X)):
        X[i] = (X[i] - mean_val)/std_val
    return X

def set_type(X, type):
    for i in range(len(X)):
        X[i] = X[i].astype(type)
    return X

def preprocess_dataset(source_path):
    # Preprocess data, ignoring compressed files and files starting with 'SA'
    i = 0
    X = []
    Y = []

    for dirName, subdirList, fileList in os.walk(source_path):
        for fname in fileList:
            if not fname.endswith('.PHN') or (fname.startswith("SA")):
                continue
            phn_fname = os.path.join(dirName, fname)
            print(phn_fname)
            wav_fname = os.path.join(dirName, fname[0:-4] + '.WAV')
            total_duration = get_total_duration(phn_fname)
            fr = open(phn_fname)
            # Generate X vector with mfccs features
            X_val, total_frames = create_mfcc(wav_fname)
            total_frames = int(total_frames)
            X.append(X_val)
            y_val = np.zeros(total_frames) - 1
            start_ind = 0
            # Generate y vector with phonetic of correlation frame
            for line in fr:
                [start_time, end_time, phoneme] = line.rstrip('\n').split()
                end_time = int(end_time)
                print(end_time)
                phoneme_num = find_phoneme(phoneme)
                end_ind = int(np.round((end_time) / total_duration * total_frames))
                print(end_ind)
                y_val[start_ind : end_ind] = phoneme_num
                start_ind = end_ind
            fr.close()

            Y.append(y_val.astype('int32'))
            i += 1
            if i >= settings.DEBUG_SIZE and settings.DEBUG:
                break
        if i >= settings.DEBUG_SIZE and settings.DEBUG:
            break

    return X, Y

def save_preprocessed_data(data):
    with open(settings.TARGET_PATH, 'wb') as cPickle_file:
        cPickle.dump(data, cPickle_file, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    print('Creating Validation index ...')
    val_idx = random.sample(range(0, settings.TRAIN_SIZE), settings.VAL_SIZE)
    val_idx = [int(i) for i in val_idx]
    print('Training set processing...')
    X_train_all, y_train_all = preprocess_dataset(settings.TRAIN_PATH)
    print('Testing set processing...')
    X_test, y_test = preprocess_dataset(settings.TEST_PATH)
    print('Separating validation and training set ...')
    X_train = []
    X_val = []
    y_train = []
    y_val = []

    for i in range(len(X_train_all)):
        if i in val_idx:
            X_val.append(X_train_all[i])
            y_val.append(y_train_all[i])
        else:
            X_train.append(X_train_all[i])
            y_train.append(y_train_all[i])

    print('Normalizing data ...')
    print('    Each channel mean=0, sd=1 ...')

    mean_val, std_val, _ = calc_norm_param(X_train)

    X_train = normalize(X_train, mean_val, std_val)
    X_val = normalize(X_val, mean_val, std_val)
    X_test = normalize(X_test, mean_val, std_val)

    X_train = set_type(X_train, 'float32')
    X_val = set_type(X_val, 'float32')
    X_test = set_type(X_test, 'float32')

    print('Saving data to {}...'.format(settings.TARGET_PATH))

    # save_preprocessed_data([X_train, y_train, X_val, y_val, X_test, y_test])

    print('Preprocessing complete!')
