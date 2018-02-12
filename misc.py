import numpy as np
import os.path
import pickle


def dump_pickle(dataset):
  data = np.genfromtxt('data/{}.csv'.format(dataset), delimiter=',', skip_header=1)
  pickle.dump(data, open('data/{}.p'.format(dataset), 'wb'))
  return data


def load_pickle(dataset):
  file_path = 'data/{}.p'.format(dataset)

  if not os.path.exists(file_path):
    return dump_pickle(dataset)

  return pickle.load(open(file_path, 'rb'))
