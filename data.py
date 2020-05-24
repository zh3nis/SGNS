import numpy as np
import torch

class DataReader:
  NEGATIVE_TABLE_SIZE = 1e8

  def __init__(self, inputFileName, min_count):

    self.negatives = []
    self.discards = []
    self.negpos = 0

    self.word2id = dict()
    self.id2word = dict()
    self.token_count = 0
    self.word_frequency = dict()

    self.inputFileName = inputFileName
    self.read_words(min_count)
    self.initTableNegatives()
    self.initTableDiscards()

  def read_words(self, min_count):
    print("Reading data", end=" ")
    word_frequency = dict()
    word_sequence = open(self.inputFileName, encoding="utf8").read().replace("\n", " ").split()

    for word in word_sequence:
      if len(word) > 0:
        self.token_count += 1
        word_frequency[word] = word_frequency.get(word, 0) + 1

        if self.token_count % 1000000 == 0:
          print(".", end="")
    print("\nTotal tokens: " + str(self.token_count))

    wid = 0
    for w, c in word_frequency.items():
      if c < min_count:
          continue
      self.word2id[w] = wid
      self.id2word[wid] = w
      self.word_frequency[wid] = c
      wid += 1
    print("Vocabulary size: " + str(len(self.word2id)))

  def initTableDiscards(self):
    t = 0.0001
    f = np.array(list(self.word_frequency.values())) / self.token_count
    self.discards = np.sqrt(t / f) + (t / f)

  def initTableNegatives(self):
    pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
    words_pow = sum(pow_frequency)
    ratio = pow_frequency / words_pow
    count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
    for wid, c in enumerate(count):
      self.negatives += [wid] * int(c)
    self.negatives = np.array(self.negatives)
    np.random.shuffle(self.negatives)

  def getNegatives(self, target, size):
    response = self.negatives[self.negpos:self.negpos + size]
    self.negpos = (self.negpos + size) % len(self.negatives)
    if len(response) != size:
      return np.concatenate((response, self.negatives[0:self.negpos]))
    return response


class Word2vecDataset(torch.utils.data.Dataset):
  def __init__(self, data, window_size, neg_num):
    self.data = data
    self.window_size = window_size
    self.neg_num = neg_num
    words = open(data.inputFileName, encoding="utf8").read().replace("\n", " ").split()
    self.word_ids = [self.data.word2id[w] for w in words if
                     w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
    self.data_len = len(self.word_ids)
    
  def __len__(self):
    return self.data_len

  def __getitem__(self, idx):
    boundary = np.random.randint(1, self.window_size + 1)
    u = self.word_ids[idx]
    return [(u, v, self.data.getNegatives(v, self.neg_num)) for v in 
            self.word_ids[max(idx - boundary, 0) : idx] + self.word_ids[min(idx + 1, self.data_len) : min(idx + boundary + 1, self.data_len)]]

  @staticmethod
  def collate(batches):
    all_u = [u for batch in batches for u, _, _ in batch]
    all_v = [v for batch in batches for _, v, _ in batch]
    all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch]

    return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)


class ValidDataset(torch.utils.data.Dataset):
  def __init__(self, train_data, valid_file, window_size, neg_num):
    self.train_data = train_data
    self.window_size = window_size
    self.neg_num = neg_num
    words = open(valid_file, encoding="utf8").read().replace("\n", " ").split()
    self.word_ids = [self.train_data.word2id[w] for w in words if
                     w in self.train_data.word2id and 
                     np.random.rand() < self.train_data.discards[self.train_data.word2id[w]]]
    self.data_len = len(self.word_ids)
    
  def __len__(self):
    return self.data_len

  def __getitem__(self, idx):
    boundary = np.random.randint(1, self.window_size + 1)
    u = self.word_ids[idx]
    return [(u, v, self.train_data.getNegatives(v, self.neg_num)) for v in 
            self.word_ids[max(idx - boundary, 0) : idx] + self.word_ids[min(idx + 1, self.data_len) : min(idx + boundary + 1, self.data_len)]]

  @staticmethod
  def collate(batches):
    all_u = [u for batch in batches for u, _, _ in batch]
    all_v = [v for batch in batches for _, v, _ in batch]
    all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch]

    return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)