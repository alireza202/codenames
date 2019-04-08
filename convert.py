#!/usr/bin/python

import re
import sys
import numpy as np
import pandas as pd

# Based on the preprocessing in https://github.com/FALCONN-LIB/FALCONN

matrix = []
words = []
with open('dataset/glove.6B.300d.txt', 'r') as inf:
    for counter, line in enumerate(inf):
        word, *rest = line.split()
        word = word.lower()
        if word.isalpha():
            words.append(word)
            row = list(map(float, rest))
            assert len(row) == 300
            vector = np.array(row, dtype=np.float32)
            vector /= np.linalg.norm(vector)
            matrix.append(vector)
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)

np.save('dataset/glove.6B.300d', np.array(matrix))

with open('dataset/words', 'w') as ouf:
    ouf.write('\n'.join(words))



# col_names = ['word']
# col_names.extend([str(i) for i in range(1, 301)])

# df = pd.read_csv(
#     'dataset/glove.6B.300d.txt', 
#     sep=' ', 
#     quoting=3, 
#     names=col_names
# )

# # getting rid of tokens with non-characters
# df = df[df['word'].str.contains(r'\w', na=False)]
# df.shape
# # (399852, 301)

# mt = df.T.to_dict('list')

# for k, v in mt.items():
#     k = np.array('v')


# with open('dataset/glove.6B.300d.pickle', 'wb') as handle:
#     pickle.dump(mt, handle, protocol=pickle.HIGHEST_PROTOCOL)

# df.to_pickle('dataset/glove.6B.300d.pickle')





