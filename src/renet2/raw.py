# coding=utf-8
from renet2.utils.tokenizer import tokenize
from renet2.utils.ann_utils import *
from renet2.utils.sequence_utils import *
import numpy as np
import pandas as pd
import sys
import os
#import tokenization
import re
from pathlib import Path

def get_token_offset(tokens, sentences):
    token_offsets = []
    for i, sentence in enumerate(sentences):
        sentence_token_offsets = []
        sentence_tokens = tokens[i]
        for j in range(len(sentence_tokens)):
            if j == 0:
                tmp_off = sentence.find(sentence_tokens[j])
                token_offset = tmp_off if tmp_off >= 0 else 0
            else:
                offset_begin = last_token_offset + len(sentence_tokens[j-1])
                sentence_to_find = sentence[offset_begin:]
                additional_offset = sentence_to_find.find(sentence_tokens[j])
                if additional_offset < 0:
                    additional_offset = 0
                token_offset = offset_begin + additional_offset
                #print(sentence_tokens[j], token_offset)
            sentence_token_offsets.append(token_offset)
            last_token_offset = token_offset
        token_offsets.append(sentence_token_offsets)
    return token_offsets

def get_sent_offset(sentences, text):
    sent_offset = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            offset = 0
        else:
            offset = last_offset + len(last_sentence)
        if sentence != text[offset: offset+len(sentence)]:
            tmp_off = text[offset:].find(sentence)
            offset = offset + (tmp_off if tmp_off >= 0 else 0)
        sent_offset.append(offset)        
        last_offset = offset
        last_sentence = sentence
    return sent_offset

def mapping_ori_tok_id_ft(doc_word_seq, doc_fixed_features, n_tokenizer, id_map=None):
    doc_word_seq_filter = []
    doc_tk_id_filter = []
    doc_features_filter = []

    for doc_no in range(len(doc_word_seq)):
        word_seq = doc_word_seq[doc_no]
        fixed_features = doc_fixed_features[doc_no]

        word_seq_filter = []
        features_filter = []
        tk_id_filter = []

        for sentence_no in range(len(word_seq)):
            sent_word_seq = word_seq[sentence_no]
            sent_features = fixed_features[sentence_no]


            tar_tk = []
            for token_no in range(len(sent_word_seq)):
                token = sent_word_seq[token_no]
                #feature = sent_features[token_no][0]
                feature = sent_features[token_no]

                t_token = ""
                if isinstance(token, tuple):
#                     print(token, feature)
                    prex_s, suf_s = "", ""
                    if feature == 1:
                        prex_s, suf_s = "[D1]", "[eD1]"
                    elif feature == 2:
                        prex_s, suf_s = "[G1]", "[eG1]"
                    elif feature == 3:
                        prex_s, suf_s = "[M1]", "[eM1]"
                    elif feature == 4:
                        prex_s, suf_s = "[D2]", "[eD2]"
                    elif feature == 5:
                        prex_s, suf_s = "[G2]", "[eG2]"
                    elif feature == 6:
                        prex_s, suf_s = "[M2]", "[eM2]"
                    else:
                        print('error', feature, token)


                    #if feature in [4, 5, 6]:
                    #    prex_s, suf_s = "#", "@"
                    
                    
                    t_token += prex_s + " "
                    if id_map:
#                         print(token, id_map)
                        if '-' in token[0][0]:
                            #t_token += " ".join(["[K" + str(id_map[tid]) + "]" for tid in token[0][0].split('-')])
                            t_token += " ".join(["$@" + tid + "$@" for tid in token[0][0].split('-')])
                            #t_token += " ".join([tid for tid in token[0][0].split('-')])
#                             t_id1, t_id2 = token[0][0].split('-')
# #                             print(t_id1, t_id2, id_map)
#                             t_token += "[K" + str(id_map[t_id1]) + "]" + " "
#                             t_token += "[K" + str(id_map[t_id2]) + "]" + " "
                        else:
                            #t_token += "[K" + str(id_map[token[0][0]]) + "]" + " "
                            t_token += "$@" + token[0][0] + "$@" 
                            #t_token += token[0][0] 

                    #t_token += " ".join([_s.lower() for _s in token[1]])
                   # t_token += " " + suf_s
                else:
                    t_token = token
                tar_tk.append(t_token)   



            _ids, _tokens, _ft = n_tokenizer.cst_encode_get_f(' '.join(tar_tk))

            tk_id_filter.append(_ids)
            word_seq_filter.append(_tokens)
            features_filter.append(_ft)

        doc_features_filter.append(features_filter)
        doc_word_seq_filter.append(word_seq_filter)
        doc_tk_id_filter.append(tk_id_filter)
    return doc_word_seq_filter, doc_tk_id_filter, doc_features_filter

def mapping_ori_tokenizer(doc_word_seq, doc_fixed_features, id_map=None):
    doc_word_seq_filter = []
    for doc_no in range(len(doc_word_seq)):
        word_seq = doc_word_seq[doc_no]
        fixed_features = doc_fixed_features[doc_no]
        word_seq_filter = []
        for sentence_no in range(len(word_seq)):
            sent_word_seq = word_seq[sentence_no]
            sent_features = fixed_features[sentence_no]
            sent_word_seq_filter = []
            for token_no in range(len(sent_word_seq)):
                token = sent_word_seq[token_no]
                feature = sent_features[token_no][0]

                t_token = ""
                if isinstance(token, tuple):
#                     print(token, feature)
                    prex_s, suf_s = "", ""
                    if feature == 1:
                        prex_s, suf_s = "[D1]", "[eD1]"
                    elif feature == 2:
                        prex_s, suf_s = "[G1]", "[eG1]"
                    elif feature == 3:
                        prex_s, suf_s = "[M1]", "[eM1]"
                    elif feature == 4:
                        prex_s, suf_s = "[D2]", "[eD2]"
                    elif feature == 5:
                        prex_s, suf_s = "[G2]", "[eG2]"
                    elif feature == 6:
                        prex_s, suf_s = "[M2]", "[eM2]"
                    else:
                        print('error', feature, token)


                    #if feature in [4, 5, 6]:
                    #    prex_s, suf_s = "#", "@"
                    
                    
                    t_token += prex_s + " "
                    if id_map:
#                         print(token, id_map)
                        if '-' in token[0][0]:
                            #t_token += " ".join(["[K" + str(id_map[tid]) + "]" for tid in token[0][0].split('-')])
                            t_token += " ".join(["$@" + tid + "$@" for tid in token[0][0].split('-')])
                            #t_token += " ".join([tid for tid in token[0][0].split('-')])
#                             t_id1, t_id2 = token[0][0].split('-')
# #                             print(t_id1, t_id2, id_map)
#                             t_token += "[K" + str(id_map[t_id1]) + "]" + " "
#                             t_token += "[K" + str(id_map[t_id2]) + "]" + " "
                        else:
                            #t_token += "[K" + str(id_map[token[0][0]]) + "]" + " "
                            t_token += "$@" + token[0][0] + "$@" 
                            #t_token += token[0][0] 

                    #t_token += " ".join([_s.lower() for _s in token[1]])
                   # t_token += " " + suf_s
        
                else:
                    t_token = token
                
                sent_word_seq_filter.append(t_token)

            word_seq_filter.append(sent_word_seq_filter)
        doc_word_seq_filter.append(word_seq_filter)
    return doc_word_seq_filter

def mapping_bert_tokenizer(doc_word_seq, doc_fixed_features, tokenizer):
    doc_word_seq_filter = []
    doc_fixed_features_filter = []
    for doc_no in range(len(doc_word_seq)):
        word_seq = doc_word_seq[doc_no]
        fixed_features = doc_fixed_features[doc_no]
        word_seq_filter = []
        fixed_features_filter = []
        for sentence_no in range(len(word_seq)):
            sent_word_seq = word_seq[sentence_no]
            sent_features = fixed_features[sentence_no]
            sent_word_seq_filter = []
            sent_features_filter = []
            sent_word_seq_filter.append(u'[CLS]')
            sent_features_filter.append([0])
            for token_no in range(len(sent_word_seq)):
                token = sent_word_seq[token_no]
                feature = sent_features[token_no]

                if isinstance(token, tuple):
                    new_token = tokenizer.tokenize(' '.join(token[1]))
                    new_feature = [feature] * len(new_token)
                else:
                    new_token = tokenizer.tokenize(token)
                    new_feature = [feature] * len(new_token)
                #if len(new_token) > 1:
                #    new_feature = new_feature[:1] + [[7] for j in new_feature[1:]]

                #    #new_token = new_token[:1]
                #    #new_feature = new_feature[:1] 

                #    #if feature[0] > 0:
                #    #    new_token = ['[unused'+str(feature[0])+']']
                #    #    new_feature = new_feature[:1] 

                ##print(token, new_token)
                
                sent_word_seq_filter.extend(new_token)
                sent_features_filter.extend(new_feature)
            sent_word_seq_filter.append(u'[SEP]')
            sent_features_filter.append([0])

            #sent_word_seq_filter = tokenizer.convert_tokens_to_ids(sent_word_seq_filter)
            word_seq_filter.append(sent_word_seq_filter)
            fixed_features_filter.append(sent_features_filter)
        doc_word_seq_filter.append(word_seq_filter)
        doc_fixed_features_filter.append(fixed_features_filter)
    return doc_word_seq_filter, doc_fixed_features_filter

def mapping_bert_token_to_id(doc_word_seq, tokenizer):
    doc_word_seq_filter = []
    for doc_no in range(len(doc_word_seq)):
        word_seq = doc_word_seq[doc_no]
        word_seq_filter = []
        for sentence_no in range(len(word_seq)):
            sent_word_seq = word_seq[sentence_no]
            sent_word_seq_filter = tokenizer.convert_tokens_to_ids(sent_word_seq)
            word_seq_filter.append(sent_word_seq_filter)
        doc_word_seq_filter.append(word_seq_filter)
    return doc_word_seq_filter

def generate_bert_f(seq_rnn, seq_len, feature):
    new_seq_rnn = []
    for doc in seq_rnn:
        new_doc = []
        for sen in doc:
            input_id = sen[:] + [0]*(seq_len-len(sen))
            input_mask = [1] * len(sen) + [0] * (seq_len-len(sen))
            input_type_ids = [0] * seq_len
            new_doc.append([input_id, input_mask, input_type_ids])
        new_seq_rnn.append(new_doc)
    new_feature = []
    for doc in feature:
        new_f = []
        for sen in doc:
            n_f = [i[0] for i in sen] + [0]*(seq_len-len(sen))
            new_f.append(n_f)
        new_feature.append(new_f)
    return new_seq_rnn, new_feature
            

def load_documents_ori(text_path, sentence_path, ner_path, word_index):
    
    documents = []
    gdas_total = []
    seqs = []
    features = []
    #abstracts file
    text_file = open(text_path, "r")
    sentence_file = open(sentence_path, "r")
    ner_file = open(ner_path, "r")
#     no_packs = 0
    pmid = ""
    
    cnt_p = 0
    while (1):
        try:
            line = text_file.readline()
            if line == '':
                break
            pmid = line.strip()

            if cnt_p >= 200:
                break
            if cnt_p % 100 == 0:
                sys.stdout.write('reading doc:#{:5d} pmid:{}\r'.format(cnt_p, pmid))
                sys.stdout.flush()

            cnt_p += 1
            title = text_file.readline().strip()
            abstract = text_file.readline().strip()
            #text = title + " " + abstract
            main_txt = text_file.readline().strip()
            text = title + " " + abstract + " " + main_txt
            text_file.readline()
            #text = abstract title + text

            sentences = []
            tokens =[]
            line = sentence_file.readline()
            if line == "\n":
                sentence_file.readline()
            while (1):
                line = sentence_file.readline()
                if line == "\n":
                    break
                sentence = line.strip()
                sentences.append(sentence)
                #split sentence into word.
                sentence_tokens = tokenize(sentence)
                tokens.append(sentence_tokens)
            #print(sentences[:3], tokens[:3])
            #if (pmid == '10022756'):
                #break
            
            anns = []
            while (1):
                line = ner_file.readline()
                if line == "\n":
                    break
                ann = line.strip().split("\t")
                ann[1] = int(ann[1])
                ann[2] = int(ann[2])
                if ';' in ann[5]:
                    ann[5] = ann[5].split(";")[1]
                ann[-1], ann[-2] = ann[-2], ann[-1]
#                 ann = process(ann)
                anns.append(ann[1:])

            #anns = [[#begin, #end, word commend, ID, type], []]
            anns = sorted(anns, key=lambda x: (x[0], x[1]))
            
            sent_offset = get_sent_offset(sentences, text)
            token_offset = get_token_offset(tokens, sentences)
#             anns = clean_anns(anns, sent_offset, text)
            #anns = change_tags(anns, sent_offset, token_offset)
            anns = change_tags(anns, sent_offset, token_offset, sentences)
#             anns = normalize_id(anns, human_genes)
            genes, diseases = seperate_genes_and_diseases(anns)
            
            ann_tag_ordered = make_tags(anns)
            word_sequence, fixed_features = generate_sequence(tokens, ann_tag_ordered)

            #filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
            #filter some symbols, and make feature to 6-d, not gene and disease to lowcase
            word_sequence, fixed_features = Filter_rnn(word_sequence, fixed_features)
            #print(word_sequence, fixed_features)

            #sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)
            if len(ann_tag_ordered[0][0]) == 6:
                sequences, feature, gdas = Generate_data_rnn_v(ann_tag_ordered, genes, diseases, word_sequence, fixed_features)
            else:
                sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)

            gdas = [[pmid] + gda for gda in gdas]
            seq_rnn = [texts_to_sequences(text, word_index) for text in sequences]
            #print(gdas)
            #print(sequences)
            #print(feature)
            #print(seq_rnn)
            #break

            seqs.extend(seq_rnn)
            features.extend(feature)
            gdas_total.extend(gdas)
                
        except (IndexError, ValueError, TypeError) as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print(e)
            print(pmid)
            continue

    gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
    
    text_file.close()
    sentence_file.close()
    ner_file.close()

    return gdas_df, seqs, features

def sentence_split(text):
    sentences = list()
    sent = ''
    piv = 0
    for idx, char in enumerate(text):
        if char in "?!":
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            else:
                sent = text[piv:idx + 1]
                piv = idx + 1

        elif char == '.':
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            elif (text[idx + 1] == ' ') and (
                    text[idx + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                sent = text[piv:idx + 1]
                piv = idx + 1
                
        if sent != '':
            sentences.append(sent)
            sent = ''

            if piv == -1:
                break

    if piv != -1:
        sent = text[piv:]
        sentences.append(sent)
        sent = ''

    return sentences


char_b_map = {'{':'}', '[':']', '(':')'}
def snt_split(text):
    snts = []
    snt = ''
    prv_p = 0
    lst_h = []
    use_F = 1
    
    b1_time = sum([1 for c in text if c in "[({"])
    b2_time = sum([1 for c in text if c in "}])"])
    if b1_time != b2_time:
        use_F = 0
    for _i, char in enumerate(text):
        if _i >= len(text) - 2:
            break
        # if using pair []{}() rule
        if use_F == 1:
            if char in "[({":
                lst_h.append((_i, char))
            if len(lst_h) > 0:
    #             if char == char_b_map[lst_h[-1][1]]:
                if char in "])}":
                    lst_h = lst_h[:-1]
#                 elif _i > lst_h[-1][0] + 300 or _i > prv_p + 300:
#                     lst_h = lst_h[:-1]
            if len(lst_h) > 0:
                continue
        if char in "!?" and (text[_i+1] == ' '):
            snt = text[prv_p: _i+1]
            prv_p = _i+1
#             print('a', char, snt,prv_p )
        elif (char in ".;"):
            IS_E = 1
            if (text[_i+1] == ' '):
                if _i > 3 and \
                    (text[_i-3:_i] == " vs" or \
                     text[_i-3:_i] == "i.e" or \
                     text[_i-3:_i] == "s.c" or \
                     text[_i-3:_i] == "e.g" or \
                     text[_i-3:_i] == " al" or \
                     text[_i-3:_i] == " Dr"):
                    IS_E = 0
                elif _i > 4 and \
                     (text[_i-4:_i] == " ref" or \
    #                   "&" in text[_i-4:_i] or \
                      text[_i-4:_i] == "e. g" or \
                      text[_i-4:_i] == " viz"):
                    IS_E = 0

                if IS_E == 1 and char == '.':
                    # leading with special char
                    if (text[_i+2] in '({[ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                        snt = text[prv_p: _i+1]
                        prv_p = _i+1
                    else:
                        _ii = _i+2
                        # lowercase combine speicial char
                        if ord(text[_ii]) >= ord('a') and ord(text[_ii]) <= ord('z'):
                            while _ii < len(text):
                                if text[_ii] in '0123456789({[ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'":
                                    snt = text[prv_p: _i+1]
                                    prv_p = _i+1
                                    break
                                elif text[_ii] == ' ':
                                    break
                                _ii += 1
                        # number combine speicial char
                        if prv_p != _i+1:
                            _ii = _i+2
                            if ord(text[_ii]) >= ord('0') and ord(text[_ii]) <= ord('9'):
                                while _ii < len(text):
                                    if text[_ii] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ|-/':
                                        snt = text[prv_p: _i+1]
                                        prv_p = _i+1
                                        break
                                    elif text[_ii] == ' ':
                                        break
    #                                 elif not (ord(text[_ii]) >= ord('0') and ord(text[_ii]) <= ord('9')):
    #                                     break
                                    _ii += 1
            # lowercase char flow '.' then, have special char "0-9, {[]" flow Uppercase
            elif _i>2 and ord(text[_i-1]) >= ord('a') and ord(text[_i-1]) <= ord('z'):
                _ii = _i+1
                while _ii < len(text) and text[_ii] in '0123456789-/()[]{}':
                    _ii += 1
                
                if _ii+1 < len(text) and text[_ii] == ' ' and \
                    text[_ii+1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    snt = text[prv_p: _i+1]
                    prv_p = _i+1
            
            if IS_E == 1 and prv_p != _i+1:
                _ii = _i+2
                while _ii < len(text):
                    if text[_ii] in ")]":
                        snt = text[prv_p: _i+1]
                        prv_p = _i+1
                        break
                    elif text[_ii] not in  " 0123456789([":
                        break
                    _ii += 1
                
#             print('b', char, snt,prv_p )
        if snt != '':
#             print('c', snt)
            snts.append(snt)
            snt = ''
    snt = text[prv_p:]
    snts.append(snt)
    return snts
            

# beware that in some case the whole para
# wrap by [], might make snt split worse
def refine_snt(text, snt_offset):
    b1_time = sum([1 for c in text if c in "[({"])
    b2_time = sum([1 for c in text if c in "}])"])
    use_F = 1 if b1_time == b2_time else 0
    
    ori_snt_s = snt_offset[:]
    new_snt_s = []
    _idx = 0
    lst_h = []
    
    for _i, char in enumerate(text):
        if use_F == 1:
            if char in "[({":
                lst_h.append((_i, char))
            if len(lst_h) > 0:
                if char in "])}":
                    lst_h = lst_h[:-1]
            
            
        if _idx < len(ori_snt_s) and ori_snt_s[_idx] == _i:
            if len(lst_h) == 0 or (len(lst_h) == 1 and char in "[({"):
                new_snt_s.append(_i)
            _idx += 1

    new_snt = []
    for _i in range(len(new_snt_s)):
        if _i == len(new_snt_s)-1:
            new_snt.append(text[new_snt_s[_i]:len(text)])
        else:
            new_snt.append(text[new_snt_s[_i]:new_snt_s[_i+1]])
#         if new_snt[-1] == '$$$':
#             new_snt = new_snt[:-1]
#         if len(new_snt[-1]) <= 10:
#             print(new_snt)
#             print(text)
#             return
    return new_snt, new_snt_s

def get_b(ori_s, tar_s):
    s = ''
    sf = 0
    for c in tar_s:
        if c in ' \t':
            if sf == 0:
                continue
            else:
                break
        if c == '(':
            sf = 1
            continue
        if c == ')':
            break
        if sf == 1 and c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            s += c
            continue
        s = ''
        break
    if len(s) <= 1:
        s = ''
    if s.lower() != ''.join([t[0] for t in ori_s.split(' ')]).lower():
        s = ''
    return s



def read_doc_sen_ann(pmid, text_file, sentence_file, ner_file, file_mode, if_read_para):
    
    title, abstract, text = '', '', ''
    # should make sure the strip process not distrupt the ann offset
    #if_read_para = 1
    #file_mode = 0
    if file_mode:
        title = text_file.readline()
        title = title[:-1] if len(title) > 0 else title
        abstract = text_file.readline()
        abstract= abstract[:-1] if len(abstract) > 0 else abstract 
        para = text_file.readline()
        para = para[:-1] if len(para) > 0 else para 
        #fix space between title, abstract, para
    
        if if_read_para == 1:
            text = title + " " + abstract + " " + para
        else:
            text = title + " " + abstract
    else:
        title = text_file.readline().strip()
        abstract = text_file.readline().strip()
        text = title + " " + abstract
    text_file.readline()
    text_len = len(text)
    
    #if pmid == '32309580':
    #    print('---')
    #    print(title)
    #    print(abstract)
    #    print(len(text)) 
    
    
    _max_snt_len = 500

    #if file_mode == 1:
    #    #_max_snt_len = 31
    #    _max_snt_len = 10

    #print("max n of sentences {}".format(_max_snt_len))
    _max_token_len = 512
    sentences = []
    tokens =[]
    line = sentence_file.readline()
    sentence_acc_len = 0
    max_text_len = 135000
    if line == "\n":
        sentence_file.readline()
        
    while (1):
        line = sentence_file.readline()
        if line == "\n":
            break
        sentence = line.strip()
        #if if_read_para == 0:
        #    if text.find(sentence) == -1:
        #        continue

        #print(sentence[:5], (text[sentence_acc_len:sentence_acc_len+5]))



        # check if sentence should save
        if len(sentences) >= _max_snt_len:
            continue

        if sentence_acc_len + len(sentence) > text_len:
            continue

        #_max_token_len = 175
        #if len(sentence) > _max_token_len:
        #    sentence = sentence[:_max_token_len]


        if sentence_acc_len +  (len(sentence) + 1) >= max_text_len:
            continue
        sentence_acc_len += (len(sentence) + 1)
        

        sentences.append(sentence)
        sentence_tokens = tokenize(sentence)
        if len(sentence_tokens) > _max_token_len:
            sentence_tokens = sentence_tokens[:_max_token_len]
        tokens.append(sentence_tokens)
    
    
#     while (1):
#         line = sentence_file.readline()
#         if line == "\n":
#             break
#         sentence = line.strip()
#         sentences.append(sentence)
        
#     snt_offset = get_sent_offset(sentences, text)
#     new_snts, new_snt_offset = refine_snt(text, snt_offset)
#     sentences = []
#     for s in new_snts:
#         if len(s) >= _max_snt_len:
#             continue
#         if sentence_acc_len + len(s) > text_len:
#             continue
#         sentence_acc_len += (len(s) + 1)
#         sentences.append(s)
        
        
        
    anns = []
    while (1):
        line = ner_file.readline()
        if line == "\n":
            break
        ann = line.strip().split("\t")
        if ';' in ann[5]:
            ann[5] = ann[5].split(";")[1]
        ann[1] = int(ann[1])
        ann[2] = int(ann[2])
    
        ann[4], ann[5] = ann[5], ann[4]
        if len(ann) > 6:
            ann[6] = int(ann[6])
        if ann[4] == 'None':
            continue
        #print(ann)
        # check if ann should save
        #if ann[1] > text_len:
        
        if ann[1] >= sentence_acc_len:
            continue
            
        #if pmid == "PMC1084346":
        #    print(ann, text[ann[1]:ann[2]])
        #if text[ann[1]:ann[2]] != ann[3]:
        #    print(ann, text[ann[1]:ann[2]])
        #if if_read_para != 1:
        #    if ann[1] >= len(text):
        #        continue
        #ann = process(ann)

        anns.append(ann[1:])
    #print(anns)
        
        
        
        
#     MAP_id = {ann[2]:ann[3:] for ann in anns}
#     mention_l = list(set([ann[2] for ann in anns]))
#     mention_l = sorted(mention_l, key = lambda x: len(x), reverse = True)
#     for ann in anns:
#         have_a_s = get_b(ann[2], text[ann[1]:])
#         if have_a_s:
#             if have_a_s not in mention_l:
#                 MAP_id[have_a_s] = MAP_id[ann[2]]
#                 mention_l.append(have_a_s)
#                 print(pmid, have_a_s)
            
            
#     new_anns = anns[:]
#     for ms in mention_l:
#         matches = re.finditer('\\b('+re.escape(ms)+')\\b', text)
# #         matches = re.finditer('\\b('+ms+')\\b', text)
#         match_l = [match.start() for match in matches]
#     #     print(ms, match_l)    
#         for m in match_l:
#             is_safe = 1
#             for ann in anns:
#                 if m >= ann[0] and m<ann[1]:
#                     is_safe = 0
#                     break
#             if is_safe:
#                 new_anns.append([m, len(ms)+m, ms] + MAP_id[ms])
                
#     anns = new_anns[:]
                
       
    anns = sorted(anns, key=lambda x: (x[0], x[1]))
    
        #ann [105, 117, 'hypogonadism', 'D007006', 'Disease']
    #using ann to tokenize
    
#     print(sentences)
#     print(anns)

    
    
#     new_snts = []
#     new_snts.extend(snt_split(title))
#     new_snts.extend(snt_split(abstract))
#     if file_mode:
#         new_snts.append(para)
#     sentences = new_snts[:]
    
#     if pmid == '10027593':
#         print('\n'.join(sentences))
#     try:
    ann_i = 0
    tokens =[]
    offset, last_offset = 0, 0
    new_anns = []
    for i, sentence in enumerate(sentences):
        snt_token = []
        if i == 0:
            offset = 0
        else:
            offset = last_offset + len(last_sentence)
        if sentence != text[offset: offset+len(sentence)]:
            tmp_off = text[offset:].find(sentence)
            if tmp_off >= 0:
                offset = offset + tmp_off
        if sentence != text[offset: offset+len(sentence)]:
            print("snt find error")
            print(sentence)
            print(text[offset: offset+len(sentence)])
        snt = sentence
#         print('s', snt)
#         print(text[offset:offset+len(snt)])

        if ann_i < len(anns) and anns[ann_i][0] >= offset and anns[ann_i][0] < offset + (len(snt)):
            lst_e = offset
            while ann_i < len(anns) and anns[ann_i][1] < offset + (len(snt) + 1):
                if ann_i > 0 and anns[ann_i-1][1] >= anns[ann_i][0]:
#                     print('ann overlap        \n', anns[ann_i-1], anns[ann_i])
                    pass
                else:
                    s_i, e_i, s_s = anns[ann_i][0], anns[ann_i][1], anns[ann_i][2]
    #                 print(anns[ann_i], len(snt), lst_e, s_i, e_i, lst_e-offset, s_i-offset, e_i-offset)
                    if s_i > lst_e:
                        snt_token.extend(tokenize(snt[lst_e-offset:s_i-offset]))
    #                     print('a', snt[lst_e-offset:s_i-offset])
                    snt_token.extend(tokenize(snt[s_i-offset:e_i-offset]))
    #                 print('b', snt[s_i-offset:e_i-offset])
    #                 print('%s|%s' % (snt[s_i-offset:e_i-offset], s_s))
                    lst_e = e_i
        #             print("--\n%s\n%s\n" % (anns[ann_i], snt))
    #                 print('%s|%s' % (sentences[snt_i][s_i-snt_acc_len:e_i-snt_acc_len], s_s))
                    if snt[s_i-offset:e_i-offset].lower() != s_s.lower():
                        print(" --- ann_token     \n%s\n%s\n" % (anns[ann_i], snt))
                        print('%s|%s' % (snt[s_i-offset:e_i-offset], s_s))
                new_anns.append(anns[ann_i])
                ann_i += 1
            if  ann_i < len(anns) and anns[ann_i][0] < offset + (len(snt)) and anns[ann_i][1] >= offset + (len(snt) + 1):
                #print(' --- ann_snt ignore     \n%s\n%s' % (sentence, anns[ann_i]))
                #print(' --- ann_snt ignore %s %s' % (pmid, anns[ann_i]))
                snt_token.extend(tokenize(snt[anns[ann_i][0]-offset:]))
                ann_i += 1
            else:
                if e_i-offset<len(snt):
                    snt_token.extend(tokenize(snt[e_i-offset:]))

#             print('c', snt[e_i-offset:])
# print('c', snt[e_i-offset:])
#             if e_i-offset < len(snt):
#                 snt_token.extend(tokenize(snt)[e_i-offset:])
#             else:
#                 print(sentences)
#                 print(anns)
#             if ann_i < len(anns) and anns[ann_i][0] >= offset and \
#                     anns[ann_i][0] < offset + (len(snt) + 1) and \
#                     anns[ann_i][1] >= offset + (len(snt) + 1):
#                 print('err')
#                 print(sentences)
#                 print(anns)
#                 print(snt)
#                 print(anns[ann_i])
        else:
#             if pmid == '10027593':
#                 print('a', len(snt), snt)
            if len(snt) > 0:
                snt_token = tokenize(snt) 
#         print(snt_token)

        tokens.append(snt_token)

        last_offset = offset
        last_sentence = sentence
    anns = new_anns[:]
    #print(anns)
#     except (IndexError, ValueError, TypeError) as e:
#             print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
 
        
    return text, sentences, tokens, anns
    

def load_documents_vis(text_path, sentence_path, ner_path, tokenizer, args=None):
    print("loading sorce info")
    #file_mode = 0, abstracts
    #file_mode = 1, full-text
    file_mode = 0
    is_using_bert_token = 0
    seq_length = 175
    
    #args.is_read_doc = 1
    args.is_using_bert_token = 0
    args.is_build_bert = False
    args.max_seq_length = 175

    if args:
        seq_length = args.max_seq_length
        file_mode = args.is_read_doc
        is_using_bert_token = args.is_using_bert_token
    
    documents = []
    gdas_total = []
    bert_features = []
    seqs = []
    all_ori_seq = {}
    features = []
    cnt_tot_s = 0
    text_file = open(text_path, "r")
    sentence_file = open(sentence_path, "r")
    ner_file = open(ner_path, "r")
#     no_packs = 0
    pmid = ""
    cnt_p = 0
    #max_doc_num = 5000
    #max_doc_num = 5000
    #max_doc_num = 200
    max_doc_num = args.max_doc_num
    #max_doc_num = -1
    if_read_para = 0 if args.read_abs else 1
    _max_token_len = 0
    
    filter_set = True
    filter_set = args.is_filter_sub
    PMC_pmid_arr = []
    if filter_set:
        label_df = pd.read_csv(args.sub_label_file)
        PMC_pmid_arr = label_df.pmid.astype(str).unique().tolist()
#     print("asd")
        
    session_file = open(text_path[:-8] + 's_' + text_path[-8:], "r")

    ner_df = None
    ori_ner_d = {}
    all_session_txt = {}
    all_sentence = {}
    all_session_map = {}
    while (1):
    
        try:
            line = text_file.readline()
            if line == '':
                break

            #read a document, from multiple files.
            pmid = line.strip()
#             print(pmid)
            text, sentences, tokens, anns = read_doc_sen_ann(pmid, text_file, sentence_file, ner_file, file_mode, if_read_para)


            # read session info
            sessions_snt = []
            sessions_h = [0, 0]
            line = session_file.readline()
            if line == "\n":
                session_file.readline()
                
            while (1):
                line = session_file.readline()
                #print(line)
                if line == "\n":
                    break
                line = line.strip()
                if len(line) > 4 and line[:4] in ['[TTL', '[ABS', '[INT', '[MET', '[RES', '[DIS', '[CON']: 
                    sessions_h[0] = line[:]
                else:
                    sessions_h[1] = line[:]
                    sessions_snt.append(sessions_h)
                    sessions_h = [0, 0]
            if sessions_h[0] != 0:
                sessions_snt.append(sessios_h)
            all_session_txt[pmid] =  sessions_snt[:]

            _session_bi = [i[0] for i in sessions_snt]
            #_session_off = get_sent_offset([i[1] for i in sessions_snt], ' '.join([i[1] for i in sessions_snt]))
            _session_off = get_sent_offset([i[1] for i in sessions_snt], text)



            all_sentence[pmid] =  sentences[:]

            #print(tokens)
            #print(sessions_snt)
            #break


            ori_ner_d[pmid] = anns          
    
    

    
            if filter_set and str(pmid) in PMC_pmid_arr:
#                 print(pmid, "in PMC")
                continue

            #if cnt_p > 10:
            #    break
            if cnt_p % 100 == 0:
                sys.stdout.write('reading doc:#{:5d} pmid:{}\r'.format(cnt_p, pmid))
                sys.stdout.flush()
            cnt_p += 1
            if max_doc_num != 0 and cnt_p > max_doc_num:
                break
                
            ##print(anns)
#             if pmid != '10365914':
#                 continue
#             print('+++', tokens)
#             print(len(anns), anns)
            
            
            #34, 73, 'gonadotropin-releasing hormone receptor', '2798', 'Gene'
            sent_offset = get_sent_offset(sentences, text)


            _session_map_l = list(np.digitize(sent_offset, _session_off) - 1)

            all_session_map[pmid] = [_session_bi, _session_map_l[:]]
#             print(tokens, sentences)
            token_offset = get_token_offset(tokens, sentences)

            if len(anns) <= 0:
                continue
            anns = clean_anns(anns, sent_offset, text)

            #for i in anns:
            #    if i[2] != text[i[0]:i[1]]:
            #        print(i, text[i[0]:i[1]])
            anns = change_tags(anns, sent_offset, token_offset, sentences)
            #print(len(anns), anns)
            if isinstance(ner_df, pd.DataFrame):
                _ner_df = pd.DataFrame(anns)
                _ner_df = _ner_df[[4, 5, 6]].rename(columns={4: 'id', 5:'type', 6:'sessions'})
                _ner_df['pmid'] = pmid
                _ner_df = _ner_df[['pmid', 'sessions', 'type', 'id']]
                _ner_df = _ner_df.sort_values(['sessions', 'type', 'id'], ascending=True).drop_duplicates()
                ner_df = pd.concat([ner_df, _ner_df])
            else:
                _ner_df = pd.DataFrame(anns)
                _ner_df = _ner_df[[4, 5, 6]].rename(columns={4: 'id', 5:'type', 6:'sessions'})
                _ner_df['pmid'] = pmid
                _ner_df = _ner_df[['pmid', 'sessions', 'type', 'id']]
                _ner_df = _ner_df.sort_values(['sessions', 'type', 'id'], ascending=True).drop_duplicates()
                ner_df = _ner_df

            genes, diseases = seperate_genes_and_diseases(anns)
            MX_id_n = 20
            d_1 = {'<'+str(j)+'>': str(i + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(genes)}
            d_2 = {'<'+str(j)+'>': str(i + MX_id_n + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(diseases)}
            id_map = {**d_1 , **d_2}
            
            #print(len(genes) * len(diseases)) 

            ann_tag_ordered = make_tags(anns)
            
            
            word_sequence, fixed_features = generate_sequence(tokens, ann_tag_ordered)
            #print(word_sequence)
            sen_len = len(word_sequence)

            #word_sequence, fixed_features = Filter_rnn(word_sequence, fixed_features)
            #print(word_sequence)
            #break

            all_ori_seq[pmid] = word_sequence
                
        except (IndexError, ValueError, TypeError) as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print(e)
            print(pmid)
            continue

    print('read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))
    
    text_file.close()
    sentence_file.close()
    ner_file.close()


    #print(gdas_df)

    #tmp_l = [14018,9958,316,13361,1723,14823,6275,6524,2147,4975,14730,3125,8801,12830,8484,4383,12453,14886,12336,7915,2864,9259,9899,7173,1450,1774,6404,13104,7878,5528,9188,11979]
    #for _i in tmp_l:
    #    print(gdas_df.iloc[_i])
    #    print(len(all_ori_seq[_i]))
    #    for _j in all_ori_seq[_i]:
    #        print(len(_j), _j)

    return all_ori_seq, ner_df.reset_index(drop=True), all_session_txt, ori_ner_d, all_sentence, all_session_map
    #return gdas_df, seqs, features


def load_documents(text_path, sentence_path, ner_path, tokenizer, args=None):
    print("load data in path {}".format(text_path))
    #file_mode = 0, abstracts
    #file_mode = 1, full-text
    file_mode = 0
    is_using_bert_token = 0
    seq_length = 175
    
    #args.is_read_doc = 1
    args.is_using_bert_token = 0
    args.is_build_bert = False
    args.max_seq_length = 175

    if args:
        seq_length = args.max_seq_length
        file_mode = args.is_read_doc
        is_using_bert_token = args.is_using_bert_token
    
    documents = []
    gdas_total = []
    bert_features = []
    seqs = []
    all_ori_seq = []
    features = []
    cnt_tot_s = 0
    text_file = open(text_path, "r")
    sentence_file = open(sentence_path, "r")
    ner_file = open(ner_path, "r")
#     no_packs = 0
    pmid = ""
    cnt_p = 0
    max_doc_num = args.max_doc_num
    if_read_para = 0 if args.read_abs else 1
    _max_token_len = 0
    
    filter_set = True
    filter_set = args.is_filter_sub
    PMC_pmid_arr = []
    if filter_set:
        label_df = pd.read_csv(args.sub_label_file)
        PMC_pmid_arr = label_df.pmid.astype(str).unique().tolist()
#     print("asd")
        
    while (1):
    
        try:
            line = text_file.readline()
            if line == '':
                break

            #read a document, from multiple files.
            pmid = line.strip()
            text, sentences, tokens, anns = read_doc_sen_ann(pmid, text_file, sentence_file, ner_file, file_mode, if_read_para)

            if len(anns) <= 0:
                continue
    
            if filter_set and str(pmid) in PMC_pmid_arr:
#                 print(pmid, "in PMC")
                continue

            if cnt_p % 100 == 0:
                sys.stdout.write('reading doc:#{:5d} pmid:{}\r'.format(cnt_p, pmid))
                sys.stdout.flush()
            cnt_p += 1
            if max_doc_num != 0 and cnt_p > max_doc_num:
                break
                
            #if '32309561' != pmid:
            #    continue
            #break
            #print('+++', tokens[13])
            #print(len(anns), anns)
            
            
            #34, 73, 'gonadotropin-releasing hormone receptor', '2798', 'Gene'
            sent_offset = get_sent_offset(sentences, text)
#             print(tokens, sentences)
            token_offset = get_token_offset(tokens, sentences)
            #print('+++', list(zip(tokens[13], token_offset[13])))

            anns = clean_anns(anns, sent_offset, text)
            #print(len(anns), anns)

            #for i in anns:
            #    if i[2] != text[i[0]:i[1]]:
            #        print(i, text[i[0]:i[1]])
            anns = change_tags(anns, sent_offset, token_offset, sentences)
            #print(len(anns), anns)

            #filter anns
            #_cnt_v = 0
            #_lst_sent_i = -1
            #n_anns = []
            #_max_snt_len = 31
            #for _i in range(len(anns)):
            #    if anns[_i][0] != _lst_sent_i:
            #        _lst_sent_i = anns[_i][0]
            #        _cnt_v += 1
            #    if _cnt_v > _max_snt_len:
            #        break
            #    n_anns.append(anns[_i])
            #anns = n_anns[:]
           
            #sys.stdout.write("doc# %s, ID: %s, len: %s, ann #: %s" % (cnt_p, pmid, len(text), len(anns)))
            #sys.stdout.flush()
            #for i in anns:
            #    print(i, tokens[i[0]][i[1]:i[2]+1])
           #     #if ' '.join(tokens[i[0]][i[1]:i[2]+1]).replace(" - ", "-") != i[3]:
           #     #print(i, tokens[i[0]][i[1]:i[2]+1])
           #     if tokens[i[0]][i[1]:i[2]+1] != tokenize(i[3]):
           #         #print(i, tokens[i[0]][i[1]:i[2]+1], tokens[i[0]])
           #         print(i, tokens[i[0]][i[1]:i[2]+1])

#          #   anns = normalize_id(anns, human_genes)
            genes, diseases = seperate_genes_and_diseases(anns)
            MX_id_n = 20
            d_1 = {'<'+str(j)+'>': str(i + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(genes)}
            d_2 = {'<'+str(j)+'>': str(i + MX_id_n + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(diseases)}
            id_map = {**d_1 , **d_2}
            
            #print(len(genes) * len(diseases)) 

            ann_tag_ordered = make_tags(anns)
            
            
            word_sequence, fixed_features = generate_sequence(tokens, ann_tag_ordered)
#             for _i in range(len(word_sequence)):
#                 for _j in range(len(word_sequence[_i])):
#                     print(word_sequence[_i][_j], sum(fixed_features[_i][_j]))

#             print(word_sequence)
#             print(fixed_features)
            sen_len = len(word_sequence)

            word_sequence, fixed_features = Filter_rnn(word_sequence, fixed_features)

            #_tmp_s_l = max([len(s) for s in word_sequence])
            #if _tmp_s_l > _max_token_len:
            #    print(len(word_sequence))
            #    _max_token_len = _tmp_s_l
            #    print(pmid)
            #    _s = ' '
            #    for s in word_sequence:
            #        if len(s) == _tmp_s_l:
            #            print(s)
            #    print(_max_token_len)

            #print(ann_tag_ordered[0])
            if len(ann_tag_ordered[0]) == 7:
                sequences, feature, gdas = Generate_data_rnn_v(ann_tag_ordered, genes, diseases, word_sequence, fixed_features)
            else:
                sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)
            #sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)
            cnt_tot_s += (len(gdas) * sen_len)
            gdas = [[pmid] + gda for gda in gdas]

            seq_rnn = None
            bert_f = []
#             if is_using_bert_token == 1:
#                 #using bert tokenizer to mapping token to id
#                 n_sequences, feature = mapping_bert_tokenizer(sequences, feature, tokenizer)
#                 seq_rnn = mapping_bert_token_to_id(n_sequences, tokenizer)
                
#                 #bert_f, feature = generate_bert_f(seq_rnn, seq_length, feature)
#                 #bert_features.extend(bert_f)
#             else:
#                 #using word_index to mapping token to id
#                 #seq_rnn = [texts_to_sequences(text, word_index) for text in sequences]
            seq_rnn = [texts_to_sequences(text, tokenizer) for text in sequences]
                
            if args.read_ori_token:
                #sequences = mapping_ori_tokenizer(sequences, feature, id_map)
                sequences, seq_rnn, feature = mapping_ori_tok_id_ft(sequences, feature, args.tokenizer, id_map)

            check_l = ['15147845']
            if cnt_p <= 2 or pmid in check_l:
            #if True:
                print("*** Example ***                      ")
                print("unique_id: %s" % (pmid))
                print('pairs', len(sequences), 'doc snt num', len(sequences[0]))
                p_idx = 0
                for _i in range(min(5, len(sequences[0]))):
                    _token = sequences[0][_i]
                    if len(_token) == 0:
                        continue
                    #n_token = n_sequences[0][0]
                    _token_id = seq_rnn[0][_i]
                    _feature = feature[0][_i]
                    print("token string: [%d] %s" % (len(_token), " ".join([str(x) for x in _token])))
                    #print("n_token string: %s" % " ".join([str(x) for x in n_token]))
                    print("token ids   : [%d] %s" % (len(_token_id), " ".join([str(x) for x in _token_id])))
                    print("fix feature : [%d] %s" % (len(_feature), " ".join([str(x) for x in _feature])))
                    p_idx += 1
                    if p_idx >= 2:
                        break
                    #if is_using_bert_token == 1:
                    #    input_mask = bert_f[0][0][1]
                    #    input_type_ids = bert_f[0][0][2]
                    #    new_features = bert_f[0][0]
                    #    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    #    print( "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
                    #    #print( "new_feature: %s" % " ".join([str(x) for x in new_features]))

            #print(len(sequences[0]))
            #_tar_s = sequences[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")
            #_tar_s = seq_rnn[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")
            #_tar_s = feature[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")


            all_ori_seq.extend(sequences)
            seqs.extend(seq_rnn)
            features.extend(feature)
            gdas_total.extend(gdas)



            #if args.raw_input_read_batch != -1:
            #    if cnt_p >= args.raw_input_read_batch:
            #        print('IN BATCH: read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))
            #        gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
            #        gdas_df.pmid = gdas_df.pmid.astype(str)
            #        gdas_df.geneId = gdas_df.geneId.astype(str)
            #        yield gdas_df, seqs, features, all_ori_seq, cnt_tot_s
            #        seqs, features, all_ori_seq, gdas_total = [], [], [], []
            #        cnt_p = 0
                
        except (IndexError, ValueError, TypeError) as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print(e)
            print(pmid)
            #break
            continue


    #if args.raw_input_read_batch != -1:
    #    if cnt_p >= 0:
    #        print('IN BATCH: read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))
    #        gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
    #        gdas_df.pmid = gdas_df.pmid.astype(str)
    #        gdas_df.geneId = gdas_df.geneId.astype(str)
    #        yield gdas_df, seqs, features, all_ori_seq, cnt_tot_s
    #        seqs, features, all_ori_seq, gdas_total = [], [], [], []
    #        cnt_p = 0
    
    print('read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))

    gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
    gdas_df.pmid = gdas_df.pmid.astype(str)
    gdas_df.geneId = gdas_df.geneId.astype(str)
    
    text_file.close()
    sentence_file.close()
    ner_file.close()


    #print(gdas_df)

    #tmp_l = [14018,9958,316,13361,1723,14823,6275,6524,2147,4975,14730,3125,8801,12830,8484,4383,12453,14886,12336,7915,2864,9259,9899,7173,1450,1774,6404,13104,7878,5528,9188,11979]
    #for _i in tmp_l:
    #    print(gdas_df.iloc[_i])
    #    print(len(all_ori_seq[_i]))
    #    for _j in all_ori_seq[_i]:
    #        print(len(_j), _j)

    return gdas_df, seqs, features, all_ori_seq, cnt_tot_s
    #return gdas_df, seqs, features

def load_documents_batch(text_path, sentence_path, ner_path, tokenizer, args=None):
    print("load data in path {}".format(text_path))
    #file_mode = 0, abstracts
    #file_mode = 1, full-text
    file_mode = 0
    is_using_bert_token = 0
    seq_length = 175
    
    #args.is_read_doc = 1
    args.is_using_bert_token = 0
    args.is_build_bert = False
    args.max_seq_length = 175

    if args:
        seq_length = args.max_seq_length
        file_mode = args.is_read_doc
        is_using_bert_token = args.is_using_bert_token
    
    documents = []
    gdas_total = []
    bert_features = []
    seqs = []
    all_ori_seq = []
    features = []
    cnt_tot_s = 0
    text_file = open(text_path, "r")
    sentence_file = open(sentence_path, "r")
    ner_file = open(ner_path, "r")
#     no_packs = 0
    pmid = ""
    cnt_p = 0
    g_cnt_p = 0
    max_doc_num = args.max_doc_num
    if_read_para = 0 if args.read_abs else 1
    _max_token_len = 0
    
    filter_set = True
    filter_set = args.is_filter_sub
    PMC_pmid_arr = []
    if filter_set:
        label_df = pd.read_csv(args.sub_label_file)
        PMC_pmid_arr = label_df.pmid.astype(str).unique().tolist()
#     print("asd")
        
    while (1):
    
        try:
            line = text_file.readline()
            if line == '':
                break

            #read a document, from multiple files.
            pmid = line.strip()
            text, sentences, tokens, anns = read_doc_sen_ann(pmid, text_file, sentence_file, ner_file, file_mode, if_read_para)

            if len(anns) <= 0:
                continue
    
            if filter_set and str(pmid) in PMC_pmid_arr:
#                 print(pmid, "in PMC")
                continue

            if cnt_p % 100 == 0:
                sys.stdout.write('reading doc:#{:5d} pmid:{}\r'.format(cnt_p, pmid))
                sys.stdout.flush()
            g_cnt_p += 1
            if max_doc_num != 0 and g_cnt_p > max_doc_num:
                break
            cnt_p += 1
                
            #if '32309561' != pmid:
            #    continue
            #break
            #print('+++', tokens[13])
            #print(len(anns), anns)
            
            
            #34, 73, 'gonadotropin-releasing hormone receptor', '2798', 'Gene'
            sent_offset = get_sent_offset(sentences, text)
#             print(tokens, sentences)
            token_offset = get_token_offset(tokens, sentences)
            #print('+++', list(zip(tokens[13], token_offset[13])))

            anns = clean_anns(anns, sent_offset, text)
            #print(len(anns), anns)

            #for i in anns:
            #    if i[2] != text[i[0]:i[1]]:
            #        print(i, text[i[0]:i[1]])
            anns = change_tags(anns, sent_offset, token_offset, sentences)
            #print(len(anns), anns)

            #filter anns
            #_cnt_v = 0
            #_lst_sent_i = -1
            #n_anns = []
            #_max_snt_len = 31
            #for _i in range(len(anns)):
            #    if anns[_i][0] != _lst_sent_i:
            #        _lst_sent_i = anns[_i][0]
            #        _cnt_v += 1
            #    if _cnt_v > _max_snt_len:
            #        break
            #    n_anns.append(anns[_i])
            #anns = n_anns[:]
           
            #sys.stdout.write("doc# %s, ID: %s, len: %s, ann #: %s" % (cnt_p, pmid, len(text), len(anns)))
            #sys.stdout.flush()
            #for i in anns:
            #    print(i, tokens[i[0]][i[1]:i[2]+1])
           #     #if ' '.join(tokens[i[0]][i[1]:i[2]+1]).replace(" - ", "-") != i[3]:
           #     #print(i, tokens[i[0]][i[1]:i[2]+1])
           #     if tokens[i[0]][i[1]:i[2]+1] != tokenize(i[3]):
           #         #print(i, tokens[i[0]][i[1]:i[2]+1], tokens[i[0]])
           #         print(i, tokens[i[0]][i[1]:i[2]+1])

#          #   anns = normalize_id(anns, human_genes)
            genes, diseases = seperate_genes_and_diseases(anns)
            MX_id_n = 20
            d_1 = {'<'+str(j)+'>': str(i + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(genes)}
            d_2 = {'<'+str(j)+'>': str(i + MX_id_n + 1) if i < MX_id_n else str(MX_id_n) for i, j in enumerate(diseases)}
            id_map = {**d_1 , **d_2}
            
            #print(len(genes) * len(diseases)) 

            ann_tag_ordered = make_tags(anns)
            
            
            word_sequence, fixed_features = generate_sequence(tokens, ann_tag_ordered)
#             for _i in range(len(word_sequence)):
#                 for _j in range(len(word_sequence[_i])):
#                     print(word_sequence[_i][_j], sum(fixed_features[_i][_j]))

#             print(word_sequence)
#             print(fixed_features)
            sen_len = len(word_sequence)

            word_sequence, fixed_features = Filter_rnn(word_sequence, fixed_features)

            #_tmp_s_l = max([len(s) for s in word_sequence])
            #if _tmp_s_l > _max_token_len:
            #    print(len(word_sequence))
            #    _max_token_len = _tmp_s_l
            #    print(pmid)
            #    _s = ' '
            #    for s in word_sequence:
            #        if len(s) == _tmp_s_l:
            #            print(s)
            #    print(_max_token_len)

            #print(ann_tag_ordered[0])
            if len(ann_tag_ordered[0]) == 7:
                sequences, feature, gdas = Generate_data_rnn_v(ann_tag_ordered, genes, diseases, word_sequence, fixed_features)
            else:
                sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)
            #sequences, feature, gdas = Generate_data_rnn(genes, diseases, word_sequence, fixed_features)
            cnt_tot_s += (len(gdas) * sen_len)
            gdas = [[pmid] + gda for gda in gdas]

            seq_rnn = None
            bert_f = []
#             if is_using_bert_token == 1:
#                 #using bert tokenizer to mapping token to id
#                 n_sequences, feature = mapping_bert_tokenizer(sequences, feature, tokenizer)
#                 seq_rnn = mapping_bert_token_to_id(n_sequences, tokenizer)
                
#                 #bert_f, feature = generate_bert_f(seq_rnn, seq_length, feature)
#                 #bert_features.extend(bert_f)
#             else:
#                 #using word_index to mapping token to id
#                 #seq_rnn = [texts_to_sequences(text, word_index) for text in sequences]
            seq_rnn = [texts_to_sequences(text, tokenizer) for text in sequences]
                
            if args.read_ori_token:
                #sequences = mapping_ori_tokenizer(sequences, feature, id_map)
                sequences, seq_rnn, feature = mapping_ori_tok_id_ft(sequences, feature, args.tokenizer, id_map)

            check_l = ['15147845']
            if (cnt_p <= 2 or pmid in check_l) and len(sequences) > 0:
            #if True:
                print("*** Example ***                      ")
                print("unique_id: %s" % (pmid))
                print('pairs', len(sequences), 'doc snt num', len(sequences[0]))
                p_idx = 0
                for _i in range(min(5, len(sequences[0]))):
                    _token = sequences[0][_i]
                    if len(_token) == 0:
                        continue
                    #n_token = n_sequences[0][0]
                    _token_id = seq_rnn[0][_i]
                    _feature = feature[0][_i]
                    print("token string: [%d] %s" % (len(_token), " ".join([str(x) for x in _token])))
                    #print("n_token string: %s" % " ".join([str(x) for x in n_token]))
                    print("token ids   : [%d] %s" % (len(_token_id), " ".join([str(x) for x in _token_id])))
                    print("fix feature : [%d] %s" % (len(_feature), " ".join([str(x) for x in _feature])))
                    p_idx += 1
                    if p_idx >= 2:
                        break
                    #if is_using_bert_token == 1:
                    #    input_mask = bert_f[0][0][1]
                    #    input_type_ids = bert_f[0][0][2]
                    #    new_features = bert_f[0][0]
                    #    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    #    print( "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
                    #    #print( "new_feature: %s" % " ".join([str(x) for x in new_features]))

            #print(len(sequences[0]))
            #_tar_s = sequences[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")
            #_tar_s = seq_rnn[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")
            #_tar_s = feature[0]
            #for _i in range(len(_tar_s)):
            #    print(len(_tar_s[_i]), _tar_s[_i])
            #print("")


            all_ori_seq.extend(sequences)
            seqs.extend(seq_rnn)
            features.extend(feature)
            gdas_total.extend(gdas)



            if args.raw_input_read_batch != -1:
                if cnt_p >= args.raw_input_read_batch:
                    print('IN BATCH [a]: read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))
                    gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
                    gdas_df.pmid = gdas_df.pmid.astype(str)
                    gdas_df.geneId = gdas_df.geneId.astype(str)
                    yield gdas_df, seqs, features, all_ori_seq, cnt_tot_s
                    seqs, features, all_ori_seq, gdas_total = [], [], [], []
                    cnt_p = 0
                
        except (IndexError, ValueError, TypeError) as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            print(e)
            print(pmid)
            #break
            continue


    if args.raw_input_read_batch != -1:
        if cnt_p > 0:
            print('IN BATCH [b]: read {} doc, max token len {} \n'.format(cnt_p, _max_token_len))
            gdas_df = pd.DataFrame(gdas_total, columns=['pmid', 'geneId', "diseaseId"])
            gdas_df.pmid = gdas_df.pmid.astype(str)
            gdas_df.geneId = gdas_df.geneId.astype(str)
            yield gdas_df, seqs, features, all_ori_seq, cnt_tot_s
            seqs, features, all_ori_seq, gdas_total = [], [], [], []
            cnt_p = 0
    

if __name__ == "__main__":
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    text_path= in_directory + "texts.txt"
    sentence_path = in_directory + "sentences.txt"
    ner_path = in_directory + "anns.txt"
    with open("model/word_index") as fp:
        word_index = cPickle.load(fp)
    
    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path, word_index)  
    label_dir = in_directory +"labels.csv"
    if os.path.exists(label_dir):
        pos_labels = pd.read_csv(label_dir)
        pos_labels.pmid = pos_labels.pmid.astype(str)
        pos_labels.geneId = pos_labels.geneId.astype(str)
        gdas = pd.merge(gdas, pos_labels, on=['pmid', "diseaseId", "geneId"], how="left")

        gdas = gdas.fillna(0)

        y = gdas.label.values

        with open(out_directory + '/y', 'wb') as fp:
            cPickle.dump(y, fp)
