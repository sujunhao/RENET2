#from collections import defaultdict
#import numpy as np
#def padding(text, Max_token=175, Max_sentence=52):
#    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
#    for i, sentence in enumerate(text):
#        for j, token in enumerate(sentence):
#            if i < Max_sentence and j < Max_token:
#                pad[i, j] = token
#    return pad
#
#def padding_feat_tag(text, Max_token=175, Max_sentence=52):
#    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
#    for i, sentence in enumerate(text):
#        for j, token in enumerate(sentence):
#            if i < Max_sentence and j < Max_token:
#                pad[i, j] = np.array(token[0])
#    return pad
#
#def data_iterator(data_X, data_X_feature, data_y=None, batch_size=32, shuffle=True):
#  # Optionally shuffle the data before training
#  dim_x, dim_y = 52, 175
#  if shuffle:
#    indices = np.random.permutation(len(data_X))
#  else:
#    indices = np.arange(len(data_X))
#
#  total_processed_examples = 0
#  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
#  for step in xrange(total_steps):
#    # Create the batch by selecting up to batch_size elements
#    batch_start = step * batch_size
#    
#    indices_batch = indices[batch_start:batch_start + batch_size]
#    
#    X = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
#    X_feature_tag = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
#    
#    y = np.empty((len(indices_batch), 1), dtype = int) if np.any(data_y) else None
#
#        # Generate data
#    for i, index in enumerate(indices_batch):
#    # Store volume
#        X[i, :, :] = padding(data_X[index])
#        X_feature_tag[i, :, :] = padding_feat_tag(data_X_feature[index])
#       
#        if data_y is not None: y[i, 0] = data_y[index]
#  
#    ###
#    yield X, X_feature_tag, y
#    total_processed_examples += len(X)
#  # Sanity check to make sure we iterated over all the dataset as intended
#  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
#
from collections import defaultdict
import numpy as np
def padding(text, Max_token=175, Max_sentence=52):
    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        for j, token in enumerate(sentence):
            if i < Max_sentence and j < Max_token:
                pad[i, j] = token
    return pad

def padding_mlt(text, Max_token=175, Max_sentence=52):
    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        sentence = sentence[0]
        for j, token in enumerate(sentence):
            if i < Max_sentence and j < Max_token:
                pad[i, j] = token
    return pad

def bert_padding_mlt(text, Max_token=175, Max_sentence=52):
    bert_f_sz = 3
    pad = np.zeros(shape=(Max_sentence, bert_f_sz, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        for z in range(bert_f_sz):
            single_s_f = sentence[z]
            for j, token in enumerate(single_s_f):
                if i < Max_sentence and j < Max_token:
                    pad[i, z, j] = token
    return pad

def padding_feat_tag(text, Max_token=175, Max_sentence=52):
    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        for j, token in enumerate(sentence):
            if i < Max_sentence and j < Max_token:
                #pad[i, j] = np.array(token[0])
                pad[i, j] = np.array(token)
    return pad


#return a batch of feature vector
def bert_data_iterator_mlt(data_X, data_X_feature, data_y=None, batch_size=32, shuffle=True, dim_x=52, dim_y=175):
  bert_f_dim = 3
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(data_X))
  else:
    indices = np.arange(len(data_X))

  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    
    indices_batch = indices[batch_start:batch_start + batch_size]
    
    #X = np.empty((len(indices_batch), dim_x, bert_f_dim, dim_y), dtype = float)
    X = np.empty((len(indices_batch), dim_x, bert_f_dim, dim_y), dtype = int)
    X_feature_tag = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    
    y = np.empty((len(indices_batch), 1), dtype = int) if np.any(data_y) else None

        # Generate data
    for i, index in enumerate(indices_batch):
    # Store volume
      #X = [[s1, sk]] k!=dim_x
      #s1 = [bert_id, bert_mask, bart_segment]
      X[i,:,:,:] = bert_padding_mlt(data_X[index], dim_y, dim_x) 
      X_feature_tag[i, :, :] = padding_feat_tag(data_X_feature[index], dim_y, dim_x)
       
      if data_y is not None: y[i, 0] = data_y[index]
  
    ###
    yield X, X_feature_tag, y
    total_processed_examples += len(X)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)

def data_iterator_mlt(data_X, data_X_feature, data_y=None, batch_size=32, shuffle=True, dim_x=52, dim_y=175):
  print(dim_x, dim_y, batch_size)
  # Optionally shuffle the data before training
  if shuffle:
    indices = np.random.permutation(len(data_X))
  else:
    indices = np.arange(len(data_X))

  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    
    indices_batch = indices[batch_start:batch_start + batch_size]
    
    X = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    X_feature_tag = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    
    y = np.empty((len(indices_batch), 1), dtype = int) if np.any(data_y) else None

        # Generate data
    for i, index in enumerate(indices_batch):
    # Store volume
        X[i, :, :] = padding_mlt(data_X[index], dim_y, dim_x) 
        X_feature_tag[i, :, :] = padding_feat_tag(data_X_feature[index], dim_y, dim_x) 
       
        if data_y is not None: y[i, 0] = data_y[index]
  
    ###
    yield X, X_feature_tag, y
    total_processed_examples += len(X)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)

def data_iterator(data_X, data_X_feature, data_y=None, batch_size=32, shuffle=True, dim_x = 52, dim_y = 175):
  # Optionally shuffle the data before training
  #dim_x, dim_y = 52, 175
  shuffle = False
  if shuffle:
    indices = np.random.permutation(len(data_X))
  else:
    indices = np.arange(len(data_X))

  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    
    indices_batch = indices[batch_start:batch_start + batch_size]
    
    X = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    X_feature_tag = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    
    y = np.empty((len(indices_batch), 1), dtype = int) if np.any(data_y) else None

        # Generate data
    for i, index in enumerate(indices_batch):
    # Store volume
        X[i, :, :] = padding(data_X[index], dim_y, dim_x)
        X_feature_tag[i, :, :] = padding_feat_tag(data_X_feature[index], dim_y, dim_x)
       
        if data_y is not None: y[i, 0] = data_y[index]
  
    ###
    yield X, X_feature_tag, y
    total_processed_examples += len(X)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)
