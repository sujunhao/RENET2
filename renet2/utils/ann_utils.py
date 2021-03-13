import numpy as np
from .omim2mesh import OMIM2MeSH
from .gene_transfer import Gene_transfer
#import cPickle
import pickle as cPickle
from .mesh_match import IsValidMeSH
import os

base_dir = os.path.dirname(__file__)
with open(base_dir + '/../resource/human_genes.txt', 'rb') as myfile:
    human_genes = cPickle.load(myfile)  

def clean_anns_in_different_sent(tags, sent_offset):
    tags_clean = []
    for i, tag in enumerate(tags):
        begin_sent_no = Sent_no(tag[0], sent_offset)
        end_sent_no = Sent_no(tag[1]-1, sent_offset)
        #if begin_sent_no == end_sent_no and len(tag) == 5:
        if begin_sent_no == end_sent_no:
            tags_clean.append(tag)
        else:
            print("clt e", tag, begin_sent_no, end_sent_no)
    return tags_clean

def Sent_no(offset, sent_offset):
    return np.digitize(offset, sent_offset).tolist()

def disambiguate_anns(tags):
    tag_to_del = []
    last_begin = -1
    last_end = -1
    last_index = -1 
    last_Id = ''
    for i, tag in enumerate(tags):
        begin = tag[0]
        end = tag[1]
        index = i
        Id = tag[-2]
        mention = tag[2]
        if end < last_end:
            if (')' in last_mention[-1:] or ('(' in last_mention and ')' not in last_mention)) and \
                ('XLI' in last_mention or 'XLRH' in last_mention or \
                'PC-1' in last_mention or 'FMF' in last_mention or 'GCA' in last_mention or 'XLA' in last_mention):
                tag_to_del.append(last_index)
            elif last_Id != 'D007153':
                tag_to_del.append(index)
            else:
                tag_to_del.append(last_index)

        if last_begin == begin and end > last_end:
            if (')' in mention[-1:] or ('(' in mention and ')' not in mention)) and \
            ('XLI' in mention or 'XLRH' in mention or \
            'PC-1' in mention or 'FMF' in mention or 'GCA' in mention or 'XLA' in mention):
                tag_to_del.append(index)
            elif Id != 'D007153':
                tag_to_del.append(last_index)
            else:
                tag_to_del.append(index)
                
        last_begin = begin
        last_end = end
        last_index = index
        last_mention = mention

    tags_clean = []
    for i, tag in enumerate(tags):
        if i not in tag_to_del:
            tags_clean.append(tag)
    return tags_clean

def longest_common_prefix(seq1, seq2):
    start = 0
    while start < min(len(seq1), len(seq2)):
        if seq1[start] != seq2[start]:
            break
        start += 1
    return seq1[:start]

def longest_common_suffix(seq1, seq2):
    return longest_common_prefix(seq1[::-1], seq2[::-1])[::-1]

def unify(tags, i, j, text, cleaned_tag_indices, prefix=True):
    i_start, i_end, i_mention, i_id, i_type = tags[i]
    j_start, j_end, j_mention, j_id, j_type = tags[j]
    if not prefix and i_end-j_end+j_start != 0 and text[i_end-j_end+j_start-1] != ' ':
        return
    tags[i][2] = j_mention    
    tags[i][-2] = j_id
    if not prefix:
        tags[i][0] = i_end-j_end+j_start
    else:
        tags[i][1] = i_start+j_end-j_start
    cleaned_tag_indices.add(i)
    
def unify_anns(tags, text):
    cleaned_tag_indices = set([])
    for i in range(len(tags)):
        for j in range(i, len(tags)):
            
            i_start, i_end, i_mention, i_id, i_type = tags[i]
            j_start, j_end, j_mention, j_id, j_type = tags[j]
            if i_id == j_id:
                continue
            if i_type == j_type and i_type == 'Disease':
                text = text.lower()
                common_prefix = longest_common_prefix(text[i_start:], text[j_start:])
                common_suffix = longest_common_suffix(text[:i_end+1], text[:j_end+1])
                
                if i_mention.lower() in common_suffix and j_mention.lower() in common_suffix:
                    if len(i_mention) <= len(j_mention) and i not in cleaned_tag_indices:
                        unify(tags, i, j, text, cleaned_tag_indices, prefix=False)
                    elif len(i_mention) > len(j_mention) and j not in cleaned_tag_indices:
                        unify(tags, j, i, text, cleaned_tag_indices, prefix=False)
                        
                elif i_mention.lower() in common_prefix and j_mention.lower() in common_prefix:
                    if len(i_mention) <= len(j_mention) and i not in cleaned_tag_indices:
                        unify(tags, i, j, text, cleaned_tag_indices, prefix=True)
                    elif len(i_mention) > len(j_mention) and j not in cleaned_tag_indices:
                        unify(tags, j, i, text, cleaned_tag_indices, prefix=True)
    return tags

def clean_anns(tags, sent_offset, text):
    #clean annotations in different sentences
    tags = clean_anns_in_different_sent(tags, sent_offset)
    return tags

    #disambiguate genes and diseases
    tags = disambiguate_anns(tags)
    
    #unify anns
    tags = unify_anns(tags, text)
    
    return tags

def normalize_id(tags):
    last_sent = -1
    last_start = -1
    last_end = -1
    last_Id = -1
    last_mention = ''
    gene_name_to_Id = {}
    gene_Id_to_index = {}

    for i, tag in enumerate(tags):
        sent, start_offset, end_offset, mention, Id = tag[0], tag[1], tag[2], str(tag[3]).lower(), tag[4]
        if tag[-1] == 'Gene':
            if sent == last_sent and start_offset == last_start and end_offset == last_end and mention == last_mention:
                continue
            if tag[3] not in gene_name_to_Id:
                gene_name_to_Id[tag[3]] = [Id]
            elif Id not in gene_name_to_Id[tag[3]]:
                gene_name_to_Id[tag[3]].append(Id)
            if Id not in gene_Id_to_index:
                gene_Id_to_index[Id] = [i]
            elif Id not in gene_Id_to_index[Id]:
                gene_Id_to_index[Id].append(i)
        last_sent = sent
        last_start = start_offset
        last_end = end_offset
        last_Id = Id
        last_mention = mention

    for gene_name in gene_name_to_Id:
        if len(gene_name_to_Id[gene_name]) > 1:
            all_human_gene = True
            all_non_human_gene = True
            for gene_Id in gene_name_to_Id[gene_name]:
                if gene_Id not in human_genes:
                    all_human_gene = False
                    non_human_gene_Id = gene_Id
                else:
                    human_gene_Id = gene_Id
                    all_non_human_gene = False
#             if not all_human_gene and all_non_human_gene:
#                 print gene_name, gene_name_to_Id[gene_name], non_human_gene_Id, gene_Id_to_index[non_human_gene_Id]
            if not all_human_gene and not all_non_human_gene:
                for non_human_gene_index in gene_Id_to_index[non_human_gene_Id]:
                    tags[non_human_gene_index][-2] = human_gene_Id

    invalid_mesh_indices = set([])
    for i, tag in enumerate(tags):    
        if tag[-1] == "Disease":
            for mesh in tag[-2].split('|'):
                if not IsValidMeSH(mesh):
                    invalid_mesh_indices.add(i)
    tags = [tag for (i, tag) in enumerate(tags) if i not in invalid_mesh_indices]
    return tags
                                
def add_dict(Id, tag, dictionary):
    if Id not in dictionary:
        dictionary[Id] = [tag]
    else:
        dictionary[Id].append(tag)

def seperate_genes_and_diseases(tags):
    genes = {}
    diseases = {}
    for tag in tags:
        new_tag = tag[:4]
        Id = tag[4]
        if Id == 'None':
            continue
        if tag[5] == "Gene":
            add_dict(Id, new_tag, genes)
        else:
            add_dict(Id, new_tag, diseases)
    return genes, diseases

def change_tags(tags, sent_offset, token_offset, sentences):
    new_tags = []
    for tag in tags:
        sent_no = Sent_no(tag[0], sent_offset) - 1
        begin_offset = tag[0] - sent_offset[sent_no]
        end_offset = tag[1] - sent_offset[sent_no] - 1
        if end_offset >= len(sentences[sent_no]):
            continue

        begin_token_no = np.digitize(begin_offset, token_offset[sent_no]).tolist() - 1
        end_token_no = np.digitize(end_offset, token_offset[sent_no]).tolist() - 1
        
        #    print(tag, sent_no, begin_offset, end_offset, len(sentences[sent_no]))
        #    print(begin_token_no, end_token_no, token_offset[sent_no])
        new_tag = [sent_no, begin_token_no, end_token_no] + tag[2:]
        #new_tag = [sent_no, begin_token_no, end_token_no] + tag[:]
        new_tags.append(new_tag)
    return new_tags

def process(ann):
    #if ann[-1][:5] == "MESH:":
    #    ann[-1] = ann[-1][5:]
        
    #if ann[-2] == "Disease":
    #    ann[-1] = OMIM2MeSH(ann[-1])
        
    #if ann[-2] == "Gene":
    #    ann[-1] = ann[-1].split(";")[0]
    #    ann[-1] = Gene_transfer(ann[-1].split('(')[0])

    if ann[-1] == "Gene":
        if ";" in ann[-2]:
            print(ann)
        ann[-2] = ann[-2].split(";")[0]
        old_a = ann[-2]
        ann[-2] = Gene_transfer(ann[-2].split('(')[0])
        new_a = ann[-2]
        if old_a != new_a:
            print(old_a, ann)

                    
    return ann

def process(_type, _id):
    if _type == "Gene":
        _id = _id.split(";")[0]
        _id = Gene_transfer(_id.split('(')[0])
    return _id 

