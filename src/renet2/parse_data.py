import sys
import os
import argparse
import pandas as pd
import time
import json 
import subprocess

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from utils.ann_utils import *


#print(os.path.dirname(os.path.realpath(__file__)))

### empty text to "$$$"
### strip each sentence
### each sentence link with a space
def refine_ann(anns, session_i = None):
    n_ann = []
    for ann in anns:
        if None in ann:
            continue
        if session_i:
            n_ann.append(ann + [session_i])
        else:
            n_ann.append(ann)
    return n_ann

def parse_ann_id(_type, _id):
    if _id == None or _id == '':
        return None
#     print(_type, _id)
    if _type == 'Disease':
        if _id[:5] == "MESH:":
            _id = _id[5:]
        _id = OMIM2MeSH(_id)
        _id = _id.split(";")[0]
    else:
        _id = _id.split(";")[0]
        _id = Gene_transfer(_id.split('(')[0])
    return _id

def read_abs_offset_f(path):
    pmid, title, abstract, anns = 0,0,0,0
    with open(path, 'r') as F:
        line = F.readline()
        if line == '':
            return (pmid, title, abstract, anns)
        pmid, _, title = line.strip().split('|')
        line = F.readline()
        tmp_l = line.strip().split('|')
        abstract = '$$$' if len(tmp_l) < 3 or len(tmp_l[2]) < 3 else tmp_l[2]

        anns = []
        line = F.readline()
        while len(line) > 1:
            ann = line.strip().split('\t')
            if len(ann) == 6:
                off_a, off_b, mention, _type, _id = ann[1:]
                if None not in ann[1:] and\
                    _type == "Disease" or _type == 'Gene':
                    _id = parse_ann_id(_type, _id)

                    anns.append([_type, _id, mention, int(off_a), int(off_b)])
                    #anns.append([pmid, off_a, off_b, mention, _type, _id])
            line = F.readline()
    return (pmid, title, abstract, anns)




# read PMID and PMCID doc
def parse_doc(doc_data, abs_f_path = None, ori_tar=None, is_SeFi = True):
    # ignore some session [abs, intro, mth, dis, etc]
    # default delete the methods session
    IG_N = 0
    if ori_tar == None:
        ori_tar = ['TITLE', 'ABSTRACT', 'INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'CONCL']
        IG_N = 3
    #IG_N = 3
    
    if '#' == ori_tar[1]:
        IG_N = 1

    EMPTY_TEXT_STR = "$$$"
    ttl_text, abs_text, para_text = '', '', ''
    pmid, pmcid = '', ''
    t_text = ''
    s_text = ''

    if doc_data:
        pmid, pmcid = doc_data['_id'].split('|')
    acc_empt_offset = 0
    anns = []

    tar_g_d_a = []
    
    s_i = 1
    
    try:
        if abs_f_path:
            pmid, ttl_text, abs_text, anns = read_abs_offset_f(abs_f_path)
            
            if IG_N == 1:
                abs_text = EMPTY_TEXT_STR
                t_ann = []
                for _a in anns:
                    # in abs
                    if _a[-1] >= len(ttl_text):
                        continue
                    t_ann.append(_a)
                anns = t_ann[:]
            
            if pmcid != '':
                s_text += pmid + ' | ' + pmcid + '\n'
            else:
                s_text += pmid + '\n'

            s_text += '[TTL]\n' + ttl_text + '\n'
            s_text += '[ABS]\n' + abs_text + '\n'

            if len(ttl_text) <= 1:
                print('error at ', pmid)

            gene_cnt = sum([1 for a in anns if a[0] == 'Disease'])
            disease_cnt = sum([1 for a in anns if a[0] == 'Gene'])
            if gene_cnt * disease_cnt > 0:
                g_i = [i[1] for i in anns if i[0] == 'Gene']
                d_i = [i[1] for i in anns if i[0] == 'Disease']
                tar_g_d_a.extend([[str(pmid), str(i), str(j)] for i in g_i for j in d_i])

            t_text = ttl_text + ' ' + abs_text
            
            s_i = 1
            if len(anns) > 0:
                anns = refine_ann(anns, s_i)
                _tmp_ann = []
                for rec in anns:
                    new_s = t_text[rec[3]:rec[4]]
                    if rec[2] != new_s:
                        print('Mis ann: {} {} |{}|, {}'.format(pmid, rec, new_s, len(new_s)))
                        continue
                    _tmp_ann.append(rec)
                anns = _tmp_ann
            s_i += 1
        
            
        if doc_data:
            for bk in doc_data['passages']:
                ty = bk['infons']['type']
                if ty != 'abstract' and ty != 'paragraph' and ty != 'front' and ty != 'title':
                    continue
        
                if 'section' in bk['infons']:
                    if bk['infons']['section'].lower() == "funding" or \
                        bk['infons']['section'].lower() == "abbreviations:" or \
                        bk['infons']['section'].lower() == "literature cited" or \
                       bk['infons']['section'].lower() == "references":
                        continue
        
        #         tar_n2 = ['METHODS', 'INTRO', 'RESULTS', 'DISCUSS', 'CONCL', 'CASE']            
                #tar_n2 = ['TITLE', 'ABSTRACT', 'INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'CONCL']
                tar_n2 = ori_tar[:]
                if IG_N > 0:
                    tar_n2[IG_N] = '#'
        #         tar_n2 = ['METHODS', 'INTRO', 'RESULTS', 'DISCUSS', 'CONCL']            
        #         tar_n2 = ['INTRO', 'INTRO', 'RESULTS', 'DISCUSS', '#']  
                if abs_f_path:
                    tar_n2 = tar_n2[2:]
                if 'section_type' in bk['infons']:
        #             print(bk['infons']['section_type'], ty)
                    if bk['infons']['section_type'] not in tar_n2:
                        continue
                    
                if len(bk['text'].lstrip()) == 0:
                    continue


                #_tar_n2 = ['TITLE', 'ABSTRACT', 'INTRO', 'METHODS', 'RESULTS', 'DISCUSS', 'CONCL']
                _tar_n2 = ori_tar[:]
                if 'section_type' not in bk['infons']:

                    section_s = bk['infons']['section'].lower()
                    if 'title' in section_s:
                        bk['infons']['section_type'] = _tar_n2[0]
                    elif 'abstract' in section_s:
                        bk['infons']['section_type'] = _tar_n2[1]
                    elif 'intro' in section_s:
                        bk['infons']['section_type'] = _tar_n2[2]
                    elif 'method' in section_s:
                        bk['infons']['section_type'] = _tar_n2[3]
                    elif 'result' in section_s:
                        bk['infons']['section_type'] = _tar_n2[4]
                    elif 'dis' in section_s:
                        bk['infons']['section_type'] = _tar_n2[5]
                    elif 'concl' in section_s:
                        bk['infons']['section_type'] = _tar_n2[6]
                    else:
                        #print('%s, %s | %s ' % (pmid, pmcid, bk['infons']['section']), end='')
                        #print("")
                        continue
            
        #         if bk['infons']['section_type'] == 'TITLE' and bk['offset'] != 0:
        #             print(pmid, bk['text'].strip())
                    
                if ('section' in bk['infons'] and bk['infons']['section'].lower()  == 'title') and \
                    bk['offset'] == 0:
                    tar_text = ttl_text
                elif ty == 'abstract':
                    tar_text = abs_text
                elif ty == 'paragraph':
                    tar_text = para_text
                    if len(abs_text) == 0:
                        abs_text = EMPTY_TEXT_STR
                        t_text += (" " + abs_text)
        #                 print(pmid, ' no abs')
                else:
                    continue
        #         print(bk['infons']['section_type'], ty)
                
                gene_cnt, disease_cnt = 0, 0
                if 'annotations' in bk:
                    gene_cnt = sum([1 for a in bk['annotations'] if a['infons']['type'] == 'Disease'])
                    disease_cnt = sum([1 for a in bk['annotations'] if a['infons']['type'] == 'Gene'])
                
                # filter sessions' paragraph if have no both Gene and Disease NER
                if is_SeFi == True:
                    if ty == 'paragraph' and (gene_cnt * disease_cnt == 0):
                        continue
        #         if (ty == 'paragraph' or ty == 'abstract') and (gene_cnt * disease_cnt == 0):
        #             continue
            
                a_text_len = len(t_text)
        
                if a_text_len > 0:
                    a_text_len += 1
                    if ty == 'paragraph' and len(para_text) > 0:
                        tar_text += ' '
                    if ty == 'abstract' and len(abs_text) > 0:
                        tar_text += ' '
                        
                acc_empt_offset = (bk['offset'] - a_text_len)
                acc_empt_offset += (len(bk['text']) - len(bk['text'].lstrip()))
                
        #         bk['text'] = bk['text'].strip()
                bk['text'] = bk['text'].strip().replace("\r", " ")
        
                tar_text += bk['text']
                t_text += (bk['text'] if len(t_text)==0 else (" "+bk['text']))
                
                if ('section' in bk['infons'] and bk['infons']['section'].lower()  == 'title') and \
                    bk['offset'] == 0:
                    ttl_text = tar_text
                elif ty == 'abstract':
                    abs_text = tar_text
        #             print(tar_text)
                elif ty == 'paragraph':
                    para_text = tar_text
                
                tmp_s = 'TITLE'
                if 'section_type' in bk['infons']:
                    tmp_s = bk['infons']['section_type']
                if tmp_s == 'TITLE':
                    tmp_s = 'TTL'
                if tmp_s == 'ABSTRACT':
                    tmp_s = 'ABS'
                s_text += '[%s|%s|%s]\n' % (tmp_s, gene_cnt, disease_cnt)
                s_text += bk['text'] + '\n'
                
                
                tmp_ann = []
                #ann.extend([(a['infons']['type'], a['infons']['identifier'], a['text'], b['offset'], b['length']) for a in bk['annotations'] for b in a['locations']])
                def par_idf(a):
                    _id = a['infons']['identifier']
                    _type = a['infons']['type']
                    return parse_ann_id(_type, _id)
        

                if 'annotations' in bk:
                    tmp_ann.extend([[str(a['infons']['type']), str(par_idf(a)), str(a['text']), b['offset']-acc_empt_offset, b['offset']-acc_empt_offset + b['length']] \
                            for a in bk['annotations'] for b in a['locations'] \
                            if (a['infons']['type'] == 'Disease' or a['infons']['type'] == 'Gene')])
        #         ann.extend([[a['infons']['type'], a['infons']['identifier'], a['text'], b['offset']-acc_empt_offset, b['length']] \
        #                     for a in bk['annotations'] for b in a['locations'] \
        #                     if (a['infons']['type'] == 'Disease' or a['infons']['type'] == 'Gene')])
        #         print('--\n', tmp_ann)
                if len(tmp_ann) > 0:
                    _tmp_ann = []
                    for rec in tmp_ann:
                        new_s = t_text[int(rec[3]):int(rec[4])]
                        if rec[2] != new_s:
                            print('Mis ann: {} {} |{}|, {}'.format(pmid, rec, new_s, len(new_s)))
                            continue
                        _tmp_ann.append(rec)
                    tmp_ann = _tmp_ann
                            
                tmp_ann = refine_ann(tmp_ann, s_i)
                s_i += 1
                if len(tmp_ann) > 0:
                    anns.extend(tmp_ann)
                if gene_cnt * disease_cnt > 0:
                    g_i = [i[1] for i in tmp_ann if i[0] == 'Gene']
                    d_i = [i[1] for i in tmp_ann if i[0] == 'Disease']
                    tar_g_d_a.extend([[str(pmid), str(i), str(j)] for i in g_i for j in d_i])
                
        #     print(anns)

        anns = refine_ann(anns)
        g_i = [i[1] for i in anns if i[0] == 'Gene']
        d_i = [i[1] for i in anns if i[0] == 'Disease']
        if (len(g_i) * len(d_i) <= 0):
            anns = []
            
        if len(abs_text) == 0:
            abs_text = EMPTY_TEXT_STR
            t_text += (" " + abs_text)
        if len(para_text) == 0:
            para_text = EMPTY_TEXT_STR
            t_text += (" " + para_text)
            
#         tar_g_d_a = pd.DataFrame(tar_g_d_a, columns=['pmid', 'geneId', 'diseaseId']).drop_duplicates().values.tolist()    
        tar_g_d_a = pd.DataFrame(tar_g_d_a).drop_duplicates().values.tolist()    
        
    except Exception as e: 
        print('Error on line {}, pmid {}'.format(sys.exc_info()[-1].tb_lineno, pmid), type(e).__name__, e)
        
    return pmid, ttl_text, abs_text, para_text, anns, t_text, s_text, tar_g_d_a

def parse_data_lst_hd(tar_id_lst, in_pmid_d, in_pmcid_d, parse_t, out_dir, if_has_s_f = True, ori_tar=None, is_SeFi = True):
    print('will parser doc at {} & {}, doc # {}, with sections {}, if have SeFi'.format(\
                                   in_pmcid_d, in_pmid_d, len(tar_id_lst), ori_tar, is_SeFi))
    start_time = time.time()

    ft_json_dir = in_pmcid_d
    abs_f_dir = in_pmid_d
    new_data_dir = out_dir

    f_doc = open(new_data_dir + "docs.txt","w")
    f_anns = open(new_data_dir + "anns.txt","w")

    s_doc = ''
    if if_has_s_f:
        s_doc = open(new_data_dir + "s_docs.txt","w")
    
    # pmcid_s, pmid_s = tar_id_lst[0]
    #print(tar_id_lst[0], len(tar_id_lst), in_pmid_d, in_pmcid_d, parse_t)
    
    all_tar_a = []
    info_l = []
    pmcid_s, pmid_s = '', ''
    hit_l, miss_l = [], [] 

    for i in range(len(tar_id_lst)): 

        if parse_t == 'ft':
            #pmcid_s, pmid_s = tar_id_lst[i]
            pmcid_s = tar_id_lst[i]
        else:
            pmid_s = tar_id_lst[i]
        print('{} {} {}'.format(i, pmid_s, pmcid_s), end = '\r')
        #print('{} {} {}'.format(i, pmid_s, pmcid_s))

        data_doc = None
        if parse_t == 'ft':
            file_path = ft_json_dir+pmcid_s
            #print(file_path)
            try:
                with open(file_path) as json_file:
                    data_doc = json.load(json_file)

                #get pmid
                #print('get')
                pmid_s, pmcid_s = data_doc['_id'].split('|')
            except:
                miss_l.append(pmcid_s)
                print('no {} records at {}'.format(pmcid_s, file_path))
                continue
    
            url_prefix = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmcids="
            url = url_prefix + pmcid_s
    #         pmc_id, ttl_text, abs_text, para_text, ann, t_text, s_text, tar_g_d_a = parse_doc(abs_f_path, data_doc)
    
        abs_f_path = abs_f_dir + pmid_s
        _pmid, ttl_text, abs_text, para_text, ann, t_text, s_text, tar_g_d_a = parse_doc(data_doc, abs_f_path, ori_tar, is_SeFi)
        if len(ann) == 0:
            print('connot find any gene-disease pairs %s %s' % (pmid_s, pmcid_s))
            miss_l.append(pmcid_s)
            continue

        if len(ttl_text) == 0:
            print('no title, ignore %s %s' % (pmid_s, pmcid_s))
            miss_l.append(pmcid_s)
            continue
        hit_l.append(pmcid_s)
        all_tar_a.extend(tar_g_d_a)
        
        ann = refine_ann(ann)
    
        #s = ttl_text
        #if s.strip () != s:
        #    print('\n{} |{}|\n|{}|\n'.format(_pmid, s[:10] + ' -- ' + s[-10:], s.strip()[:10] + ' -- ' +  s.strip()[-10:]))
        #s = abs_text
        #if s.strip () != s:
        #    print('\n{} |{}|\n|{}|\n'.format(_pmid, s[:10] + ' -- ' + s[-10:], s.strip()[:10] + ' -- ' +  s.strip()[-10:]))
        #s = para_text
        #if s.strip () != s:
        #    print('\n{} |{}|\n|{}|\n'.format(_pmid, s[:10] + ' -- ' + s[-10:], s.strip()[:10] + ' -- ' +  s.strip()[-10:]))
            
        if _pmid != pmid_s:
            print('error in PMID query', _pmid, pmid_s)
        a_text = ttl_text + '$' + abs_text + '$' + para_text
        for rec in ann:
            new_s = a_text[rec[3]:rec[4]]
            if rec[2] != new_s:
                print('----    {} |{}|, {}'.format(rec, new_s, len(new_s)))
                print(pmid_s)
                break
    #     f_doc.write("%s\n[TTL]\n%s\n[ABS]\n%s\n[PAR]\n%s\n\n" % (pmid_s, ttl_text, abs_text, para_text))
        f_doc.write("%s\n%s\n%s\n%s\n\n" % (pmid_s, ttl_text, abs_text, para_text))
        if if_has_s_f:
            s_doc.write("%s\n" % (s_text))
        
        for rec in ann:
            f_anns.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (pmid_s, rec[3], rec[4], rec[2], rec[0], rec[1], rec[-1]))
                
        f_anns.write("\n")
        
        all_g_l  = list(set([str(rec[1]) for rec in ann if rec[0]=='Gene']))
        all_d_l = list(set([str(rec[1]) for rec in ann if rec[0]=='Disease' if rec[1] != None]))
        
    #     info_l.append([i, pmid_s, pmcid_s, len(ttl_text), len(abs_text), len(para_text),  len(all_g_l), len(all_d_l)])
        info_l.append([pmid_s, pmcid_s, len(ttl_text), len(abs_text), len(para_text), \
                       sum([len(ttl_text), len(abs_text), len(para_text)]), \
                       len(all_g_l), len(all_d_l), len(all_g_l) * len(all_d_l), len(tar_g_d_a)])
          
    f_doc.close()
    if if_has_s_f:
        s_doc.close()
    f_anns.close()
    
    f_labels = open(new_data_dir + "labels.txt","w")
    f_labels.write("%s,%s,%s,%s\n" % ('pmid','geneId','diseaseId','label'))
    f_labels.write("%s,%s,%s,%s\n" % ('tmpid','tmpid','tmpid', '0'))
    f_labels.close()

    if if_has_s_f:
        print('hit {}, miss {}'.format(len(hit_l), len(miss_l)))

    print('result RENET2 model\'s input is at %s' % (new_data_dir))
    print("--- %.3f seconds ---" % (time.time() - start_time))

    print('using GENIA Sentence Splitter')
    
    start_time = time.time()
    
    def split_snt(new_data_dir):
        # geniass path
        g_path = print( os.path.dirname(os.path.realpath(__file__)))
        cmd = os.path.dirname(os.path.realpath(__file__)) + "/tools/geniass/run_geniass.sh " + \
                         new_data_dir + "docs.txt" + " " + \
                         new_data_dir + "sentences.txt " + \
                         subprocess.check_output(['which', 'ruby']).strip().decode("utf-8")
                         #"/usr/bin/ruby"
        print('using geniass', cmd)
        subprocess.check_call(cmd,   shell=True)
        
    split_snt(new_data_dir)
    
    print("--- %.3f seconds ---" % (time.time() - start_time))
    return info_l


# parse document from json file
def parse_data_json_f_hd(tar_json_f, out_dir, if_has_s_f = True):
    start_time = time.time()

    new_data_dir = out_dir

    f_doc = open(new_data_dir + "docs.txt","w", encoding='utf-8')
    f_anns = open(new_data_dir + "anns.txt","w", encoding='utf-8')

    s_doc = ''
    if if_has_s_f:
        s_doc = open(new_data_dir + "s_docs.txt","w", encoding='utf-8')
    
    
    all_tar_a = []
    info_l = []
    pmcid_s, pmid_s = '', ''
    hit_l, miss_l = [], [] 

    tar_doc_dic = []
    with open(tar_json_f, encoding='utf-8') as f:
        data = json.load(f)
        tar_doc_dic = data[1]
    print('loaded data, begin parsing')

    filter_pmid_l = ['32659830']
    for i in range(len(tar_doc_dic)): 
        data_doc = tar_doc_dic[i]
        pmid_s, ttl_text, abs_text, para_text, ann, t_text, s_text, tar_g_d_a = parse_doc(data_doc)
        #print(pmid_s, ttl_text, abs_text, para_text, ann, t_text, s_text)
        #break

        print('{} {} {}'.format(i, pmid_s, pmcid_s), end = '\r')

        if len(ttl_text) == 0:
            #print('no title, ignore %s %s' % (pmid_s, pmcid_s))
            miss_l.append(pmid_s)
            continue
        if len(ann) == 0:
            #print('no ann, ignore %s %s' % (pmid_s, pmcid_s))
            miss_l.append(pmid_s)
            continue

        if pmid_s in filter_pmid_l:
            miss_l.append(pmid_s)
            continue

        all_tar_a.extend(tar_g_d_a)
        
        ann = refine_ann(ann)
            
        a_text = ttl_text + '$' + abs_text + '$' + para_text

        for rec in ann:
            new_s = a_text[rec[3]:rec[4]]
            if rec[2] != new_s:
                print('----    {} |{}|, {}'.format(rec, new_s, len(new_s)))
                print(pmid_s)
                break
        all_g_l  = list(set([str(rec[1]) for rec in ann if rec[0]=='Gene']))
        all_d_l = list(set([str(rec[1]) for rec in ann if rec[0]=='Disease' if rec[1] != None]))
        if len(all_g_l) * len(all_d_l) <= 0:
            miss_l.append(pmid_s)
            continue



        f_doc.write("%s\n%s\n%s\n%s\n\n" % (pmid_s, ttl_text, abs_text, para_text))
        if if_has_s_f:
            s_doc.write("%s\n" % (s_text))
        
        for rec in ann:
            f_anns.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (pmid_s, rec[3], rec[4], rec[2], rec[0], rec[1], rec[-1]))
                
        f_anns.write("\n")
        
        
        info_l.append([pmid_s, pmcid_s, len(ttl_text), len(abs_text), len(para_text), \
                       sum([len(ttl_text), len(abs_text), len(para_text)]), \
                       len(all_g_l), len(all_d_l)])
        hit_l.append(pmid_s)
          
    f_doc.close()
    if if_has_s_f:
        s_doc.close()
    f_anns.close()
    
    f_labels = open(new_data_dir + "labels.txt","w")
    f_labels.write("%s,%s,%s,%s\n" % ('pmid','geneId','diseaseId','label'))
    f_labels.write("%s,%s,%s,%s\n" % ('tmpid','tmpid','tmpid', '0'))
    f_labels.close()

    if if_has_s_f:
        print('hit {}, miss {}'.format(len(hit_l), len(miss_l)))

    print('result RENET2 model\'s input is at %s' % (new_data_dir))
    print("--- %.3f seconds ---" % (time.time() - start_time))

    print('using GENIA Sentence Splitter')
    
    start_time = time.time()
    
    def split_snt(new_data_dir):
        # geniass path
        g_path = print( os.path.dirname(os.path.realpath(__file__)))
        cmd = os.path.dirname(os.path.realpath(__file__)) + "/../tools/geniass/run_geniass.sh " + \
                         new_data_dir + "docs.txt" + " " + \
                         new_data_dir + "sentences.txt " + \
                         subprocess.check_output(['which', 'ruby']).strip().decode("utf-8")
                         #"/usr/bin/ruby"
        print('using geniass', cmd)
        subprocess.check_call(cmd,   shell=True)
        
    split_snt(new_data_dir)
    
    print("--- %.4f seconds ---" % (time.time() - start_time))
    return all_tar_a
    

# tar_lst = [pmid, pmcid]
def parse_data_ft(tar_lst, in_pmid_d,  in_pmcid_d, out_dir, if_has_s_f=True):
    try:
        if len(tar_lst) > 0:
            #tar_lst = tar_lst[int(len(tar_lst)/2):]
            print('begin parsing full-text data, {}'.format(len(tar_lst)))
            parse_data_lst_hd(tar_lst, in_pmid_d, in_pmcid_d, 'ft', out_dir, if_has_s_f)
    except Exception as e: 
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
    pass

def parse_data_abs(tar_lst, in_pmid_d, out_dir, if_has_s_f=True):
    try:
        if len(tar_lst) > 0:
            print('begin parsing abstracts data, {}'.format(len(tar_lst)))
            parse_data_lst_hd(tar_lst, in_pmid_d, None, 'abs', out_dir, if_has_s_f)
    except Exception as e: 
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def parse_data(config):

    id_f = config.id_f
    is_ft = True if config.type == "ft" else False

    if config.in_abs_dir is None:
        print('Error! Parsing abstracts require abstracts pubtator files, please set the in_abs_dir first')
        return
    if is_ft:
        if config.in_ft_dir is None:
            print('Error! Parsing full-text require full-text json files, please set the in_ft_dir first')
            return
    in_abs_dir = config.in_abs_dir
    in_ft_dir = config.in_ft_dir
    out_dir = config.out_dir


    try:
        id_df = pd.read_csv(id_f, header=0, dtype=str)
        id_df = id_df.drop_duplicates()

        if is_ft:
            #id_df['pmcid'] = id_df['pmcid'].apply(lambda x: x.strip())
            #tar_lst = id_df[['pmcid', 'pmid']].values.tolist()
            tar_lst = list(id_df.pmcid.values)
            #print(tar_lst[:3])
            parse_data_ft(tar_lst, in_abs_dir, in_ft_dir, out_dir, not config.no_s_f)
        else:
            tar_lst = list(id_df.pmid.values)
            parse_data_abs(tar_lst, in_abs_dir, out_dir, not config.no_s_f)

    except Exception as e: 
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)



def main():
    print("***\nplease make sure to use 'renet2 install_geniass' to install the sentence splliter, before running the parse_data\n***\n")
    parser = argparse.ArgumentParser(description="parse abstrcts/full-text pubtator/json file to RENET2 input format")

    parser.add_argument('--id_f', type=str, default="../test/test_download_pmid_list.csv",
                        help="PMID/PMCID list file input, default: %(default)s")

    parser.add_argument('--type', type=str, default="abs",
            help="[abs, ft] download text type: abstrcts or full-text default: %(default)s")

    parser.add_argument('--in_abs_dir', type=str, default=None,
            help="input abstracts raw file dir default: %(default)s")

    parser.add_argument('--in_ft_dir', type=str, default=None,
            help="input full-text raw file dir default: %(default)s")

    parser.add_argument('--out_dir', type=str, default="../data/test_input/",
            help="output file dir default: %(default)s")

    parser.add_argument('--no_s_f', action='store_true', default=False,
                                help='disables generate the source session info file')

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    parse_data(args)


if __name__ == "__main__":
    main()
