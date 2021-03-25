#import cPickle
import pickle as cPickle
import os

base_dir = os.path.dirname(__file__)
with open(base_dir + '/../resource/dict_c2d.txt', 'rb') as myfile:
    dict_c2d = cPickle.load(myfile)  
    
def IsValidMeSH(Id):
    if Id[0] == "D": return True
    if Id[0] == "C":
        if dict_c2d.get(Id) is not None:
            return True
    return False
