import os
import pickle
import numpy as np
from scipy import spatial
import sys


model_path = './models/'
loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

'''Extract word pairs from word_analogy_dev.txt and store in word_analogy_{cost}.txt'''

def get_pair_embedding_diff(pair, embeddings, dictionary):
    pairs = pair.split(':')
    pairA = embeddings[dictionary[pairs[0].strip('"')]]
    pairB = embeddings[dictionary[pairs[1].strip('"')]]
    return np.subtract(pairA, pairB);


with open('word_analogy_dev.txt') as read_file:
    with open('word_analogy_%s.txt'%loss_model , 'w') as out_file:
        for in_line in read_file:
            in_line = in_line.strip()
            examples_list = in_line.split('||')[0].split(',')
            examples_list_diff = []
            for pairs in examples_list:
                examples_list_diff.append(get_pair_embedding_diff(pairs, embeddings, dictionary))

            examples_list_diff = np.mean(examples_list_diff, axis=0)

            choices_list = in_line.split('||')[1].split(',')
            max_val = - sys.maxsize - 1;
            min_val =  sys.maxsize
            most_illustrative_pair = ""
            least_illustrative_pair = ""
            for word_pair in choices_list:
                diff = get_pair_embedding_diff(word_pair,embeddings,dictionary)
                word_similarity = np.sum(np.multiply(examples_list_diff, diff)) / np.multiply(np.linalg.norm(examples_list_diff), np.linalg.norm(diff))
                if word_similarity < min_val:
                    min_val = word_similarity
                    least_illustrative_pair = word_pair
                if word_similarity > max_val:
                    max_val = word_similarity
                    most_illustrative_pair = word_pair
            output_line = in_line.split('||')[1].replace(",", " ") + " " + least_illustrative_pair + " " + most_illustrative_pair + "\n"
            out_file.write(output_line)