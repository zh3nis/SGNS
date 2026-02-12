import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


parser = argparse.ArgumentParser(description='Evaluate Word Embeddings')
parser.add_argument('--emb', type=str, default='embeddings/sgns',
                    help='path to word embeddings')
args = parser.parse_args()


wv_from_text = KeyedVectors.load_word2vec_format(args.emb, binary=False)
ws353 = wv_from_text.evaluate_word_pairs(datapath('wordsim353.tsv'))
print('WS353 = %.3f' % ws353[0][0], end=', ')
men = wv_from_text.evaluate_word_pairs('testsets/bruni_men.txt')
print('MEN = %.3f' % men[0][0], end=', ')
mturk = wv_from_text.evaluate_word_pairs('testsets/radinsky_mturk.txt')
print('M. Turk = %.3f' % mturk[0][0], end=', ')
rare = wv_from_text.evaluate_word_pairs('testsets/luong_rare.txt')
print('Rare = %.3f' % rare[0][0], end=', ')
google = wv_from_text.evaluate_word_analogies(datapath('questions-words.txt'))
print('Google = %.3f' % google[0], end=', ')
msr = wv_from_text.evaluate_word_analogies('testsets/msr.txt')
print('MSR = %.3f' % msr[0])