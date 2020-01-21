import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath


parser = argparse.ArgumentParser(description='Evaluate Word Embeddings')
parser.add_argument('--emb', type=str, default='embeddings/sgns',
                    help='path to word embeddings')
args = parser.parse_args()


wv_from_text = KeyedVectors.load_word2vec_format(args.emb, binary=False)
ws353 = wv_from_text.evaluate_word_pairs(datapath('wordsim353.tsv'))
men = wv_from_text.evaluate_word_pairs('testsets/bruni_men.txt')
mturk = wv_from_text.evaluate_word_pairs('testsets/radinsky_mturk.txt')
rare = wv_from_text.evaluate_word_pairs('testsets/luong_rare.txt')
google = wv_from_text.evaluate_word_analogies(datapath('questions-words.txt'))
msr = wv_from_text.evaluate_word_analogies('testsets/msr.txt')
print('WS353 = %.3f' % ws353[0][0], end=', ')
print('MEN = %.3f' % men[0][0], end=', ')
print('M. Turk = %.3f' % mturk[0][0], end=', ')
print('Rare = %.3f' % rare[0][0], end=', ')
print('Google = %.3f' % google[0], end=', ')
print('MSR = %.3f' % msr[0])