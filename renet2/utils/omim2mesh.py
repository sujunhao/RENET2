#import cPickle
import pickle as cPickle
import os

base_dir = os.path.dirname(__file__)  
with open(base_dir + '/../resource/dict_omim2mesh.txt', 'rb') as myfile:
    dict_omim2mesh = cPickle.load(myfile)
def OMIM2MeSH(Id):
    if Id[:4] != 'OMIM':
        return Id
    OMIM = Id.split('OMIM:')[1]
    if OMIM not in dict_omim2mesh:
        return Id
    else:
        return '|'.join(dict_omim2mesh[OMIM])
