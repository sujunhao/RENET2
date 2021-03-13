#import cPickle
import pickle as cPickle
import os

base_dir = os.path.dirname(__file__)

with open(base_dir + '/../resource/dict_gene.txt', 'rb') as myfile:
    dict_gene = cPickle.load(myfile)  
    
def Gene_transfer(species_gene):
    if species_gene not in dict_gene:
        return species_gene
    return '|'.join(dict_gene[species_gene])
