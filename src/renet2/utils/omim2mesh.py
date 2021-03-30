#import cPickle
import pickle as cPickle
import os
from pkg_resources import resource_string, resource_filename 

#base_dir = os.path.dirname(__file__)  
#file_n = base_dir + '/../resource/dict_omim2mesh.txt'
file_n = resource_filename(__name__, '/../resource/dict_omim2mesh.txt')

with open(file_n, 'rb') as myfile:
    dict_omim2mesh = cPickle.load(myfile)
def OMIM2MeSH(Id):
    if Id[:4] != 'OMIM':
        return Id
    OMIM = Id.split('OMIM:')[1]
    if OMIM not in dict_omim2mesh:
        return Id
    else:
        return '|'.join(dict_omim2mesh[OMIM])
