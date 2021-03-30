
def make_tags(tags):
    tag_duplicates = []
    last_sent = -1
    last_start = -1
    last_end = -1
    last_type = ''
    last_Id = -1
    last_mention = ''
    for i, tag in enumerate(tags):
        tag_s_i = 0
        if len(tag) == 7:
            tag_s_i = tag[6]
            tag = tag[:6]
            
        sent, start_offset, end_offset, tag_type, mention, Id = tag[0], tag[1], tag[2], tag[-1], str(tag[3]).lower(), tag[-2]
        if sent == last_sent and end_offset == last_end and start_offset == last_start and mention == last_mention:
#             start_tokens = [token for token in abstract_tokens_normal_clean[pubmed][sent][start_offset:end_offset+1]]
#             end_tokens = abstract_tokens[pubmed][sent][last_start:last_end+1]
            if last_tag != '' and last_tag_type != tag_type:
                tag[-1] = 'Gene-Disease'
                if tag_type == 'Gene':
                    tag[-2] = str(last_Id) + '>-<' + str(Id)
                else:
                    tag[-2] = str(Id) + '>-<' + str(last_Id)
                tag_duplicates.append(i-1)
        last_sent = sent
        last_start = start_offset
        last_end = end_offset
        last_tag_type = tag_type
        last_Id = Id
        last_tag = tag
        last_mention = mention
    tags = [tags[i] for i in range(len(tags)) if i not in tag_duplicates]
    
    new_tags = []
    last_sent = -1
    last_start = -1
    last_end = -1
    last_tag = ''
    last_Id = -1
    last_tag_name = ''
    for tag in tags:
        tag_s_i = 0
        if len(tag) == 7:
            tag_s_i = tag[6]
            tag = tag[:6]

        sent, start_offset, end_offset, mention, Id, tag_type, tag_name = \
            tag[0], tag[1], tag[2], str(tag[3]).lower(), tag[4], tag[-1], '<' + tag[-2] + '>'
        if sent == last_sent and start_offset <= last_end and last_tag != '':
            new_tag_name = tag_name
            new_tag_type = tag_type
            #if last_tag_name == tag_name:
            #    new_tag_name = tag_name
            #    new_tag_type = tag_type
            #elif type(last_tag_name) == list:
            #    new_tag_name = last_tag_name + [tag_name]
            #    new_tag_type = last_tag_type + [tag_type]
            #else:
            #    new_tag_name = [last_tag_name, tag_name]
            #    new_tag_type = [last_tag_type, tag_type]
            if tag_s_i != 0:
                new_tag = [sent, last_start, end_offset, mention, new_tag_name, new_tag_type, tag_s_i]
            else:
                new_tag = [sent, last_start, end_offset, mention, new_tag_name, new_tag_type]

            del new_tags[-1]
            new_tags.append(new_tag)
        else:
            if tag_s_i != 0:
                new_tags.append(tag[:-2] + [tag_name] + [tag[-1], tag_s_i])
            else:
                new_tags.append(tag[:-2] + [tag_name] + [tag[-1]])

        last_tag = new_tags[-1][:6]
        last_sent, last_start, last_end, last_mention, last_Id, last_tag_type, last_tag_name = \
            last_tag[0], last_tag[1], last_tag[2], str(last_tag[3]).lower(), last_tag[4], last_tag[-1], last_tag[-2]
    
    return new_tags

def generate_sequence(sentences, tags, _have_snt_no_tab=True):
#     for i in sentences:
#         print(i)
#     for i in tags:
#         print(i)
#     print(" ")
    #_have_snt_no_tab = False

    new_sentences = []
    sent_features = []
    tag_no = 0
    for sentence_no in range(len(sentences)):
        sentence = sentences[sentence_no]
        if tag_no == len(tags):
            if _have_snt_no_tab:
                new_sentences.append(sentence)
                sent_features.append([[0, 0, 0, 0]] * len(sentence))
            continue
        tag = tags[tag_no]
        FLAG = 0
        
        if tag[0] == sentence_no:
            last_end = 0
            new_sentence = []
            sent_feature = []
#             print(sentence)
#             print(tag)
#             print(" ")
            FLAG = 0
            while tag[0] == sentence_no:
                start = tag[1]
                end = tag[2] + 1
                tmp_feature = []
                if type(tag[4]) == list:
                    tag_name = tag[4]
                    feature = [One_hot_feature(tag_type) for tag_type in tag[5]]
                    if len(feature) > 0:
                        FLAG = 1
                        tmp_feature = feature[0][:]
                        for i in range(4):
                            tmp_feature[i] = 1 if sum([j[i] for j in feature]) > 0 else 0
                        feature = [tmp_feature[:]]
                    new_sentence += sentence[last_end:start] + [(tag_name, sentence[start:end])]
#                     new_sentence += sentence[last_end:start] + [(j, sentence[start:end]) for j in tag_name]
                    sent_feature += [[0, 0, 0, 0]] * len(sentence[last_end:start]) + feature[:1]
                else:
                    tag_name = [tag[4]]
                    feature = [One_hot_feature(tag[5])]
#                 if sentence_no == 1:
#                     print(tag_name)
#                     print('---', feature, tmp_feature)
#                     print(sentence[last_end:start])
#                     print((sentence[start:end]))
#                     print(new_sentence)
#                     print(sent_feature)
#                     print("--")
                    new_sentence += sentence[last_end:start] + [(tag_name, sentence[start:end])]
                    sent_feature += [[0, 0, 0, 0]] * len(sentence[last_end:start]) + feature
#                 new_sentence += sentence[last_end:start] + [(tag_name, sentence[start:end])]
#                 sent_feature += [[0, 0, 0, 0]] * len(sentence[last_end:start]) + feature
#                 if sentence_no == 1:
#                     print(new_sentence)
#                     print(sent_feature)
#                     print("--")
                tag_no += 1
                last_end = end
                if  tag_no < len(tags):
                    tag = tags[tag_no]
                else:
                    break
            new_sentence += sentence[end:]
            sent_feature += [[0, 0, 0, 0]] * len(sentence[end:])

            new_sentences.append(new_sentence)
            sent_features.append(sent_feature)
#             if FLAG==1:
                
# #                 def f(x) = lambda x: ''.join([str(t) for t in x])
#                 print(len(new_sentence), new_sentence)
#                 print(len(sent_feature), sent_feature)
        else:
            if _have_snt_no_tab:
                new_sentences.append(sentence)
                sent_features.append([[0, 0, 0, 0]] * len(sentence))
            continue
        
    return new_sentences, sent_features

def Filter_rnn(word_seq, fixed_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    
    word_seq_filter = []
    fixed_features_filter = []
    for sentence_no in range(len(word_seq)):
        sent_word_seq = word_seq[sentence_no]
        sent_features = fixed_features[sentence_no]
        sent_word_seq_filter = []
        sent_features_filter = []
        for token_no in range(len(sent_word_seq)):
            token = sent_word_seq[token_no]
            feature = sent_features[token_no]
            if isinstance(token, tuple):
                sent_word_seq_filter.append(token)
                sent_features_filter.append(feature[:4] + feature[-2:])
#                 if sum(sent_features_filter[-1]) == 0:
#                     print('-------------------------')
#                     print(sent_word_seq_filter)
#                     print([sum(i) for i in sent_features_filter])
            else:
                if token not in filters:
                    if sum(feature[:4]) == 0:
                        sent_word_seq_filter.append(token.lower())
                    else:
                        sent_word_seq_filter.append(token)
                    sent_features_filter.append(feature[:4] + feature[-2:])
        word_seq_filter.append(sent_word_seq_filter)
        fixed_features_filter.append(sent_features_filter)
    return word_seq_filter, fixed_features_filter

def One_hot_feature(tag_type):
    if tag_type == "Disease":
        return [1, 0, 0, 0]
    elif tag_type == 'Gene':
        return [0, 1, 0, 0]
    elif tag_type == 'Gene-Disease':
        return [1, 1, 0, 0]
    
def texts_to_sequences_generator(texts, word_index):
    """Transforms each text in texts in a sequence of integers.
    Only top "num_words" most frequent words will be taken into account.
    Only words known by the tokenizer will be taken into account.
    # Arguments
        texts: A list of texts (strings).
    # Yields
        Yields individual sequences.
    """
#     num_words = self.num_words
    for seq in texts:
        vect = []
        for w in seq:
            if isinstance(w, tuple):
                w = w[0][0]
            i = word_index.get(w)
            if i is not None:
                vect.append(i)
            else:
                #vect.append(word_index['UUUNKKK'])
                #if len(w) >= 3 and w[0] == '<' and w[-1] == '>':
                #    if w[1] == 'D':
                #        vect.append(word_index["<UNKNOWN_DISEASE>"])
                #    elif ord(w[1]) >= 48 and ord(w[1]) <= 57:    #is number
                #        vect.append(word_index["<UNKNOWN_GENE>"])
                #    else:
                #        vect.append(word_index["[X]"])
                #else:
                #    vect.append(word_index["[X]"])
                vect.append(word_index["[X]"])
        yield vect

def texts_to_sequences(texts, word_index):
        """Transforms each text in texts in a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            texts: A list of texts (strings).
        # Returns
            A list of sequences.
        """
        res = []
        for vect in texts_to_sequences_generator(texts, word_index):
            res.append(vect)
        return res
    
# check if a gene/disease pairs inside a session
def Generate_data_rnn_v(anns, genes, diseases, word_sequences, fixed_features):

    all_p_l = []
    tmp_s_ann = []
    lst_s = 0
    anns = sorted(anns, key=lambda x: (int(x[-1]), (int(x[0]), int(x[1]))))
    for ann in anns:
        if len(tmp_s_ann) > 0 and ann[-1] != lst_s:
            # adding pairs

            g_l = [i[4].split('-')[0][1:-1] for i in tmp_s_ann if i[5] == 'Gene']
            d_l = []
            try:
                d_l = [i[4].split('-')[0][1:-1] for i in tmp_s_ann if i[5] != 'Gene']
            except:
                print('warning', tmp_s_ann)
            if len(g_l) * len(d_l) > 0:
                all_p_l.extend([(i, j) for i in g_l for j in d_l])
                #print([(i, j) for i in g_l for j in d_l])
            tmp_s_ann = []
            lst_s = 0
        if len(tmp_s_ann) == 0 or ann[-1] == lst_s:
            tmp_s_ann.append(ann)
            lst_s = ann[-1]

    if len(tmp_s_ann) > 0:
        g_l = [i[4].split('-')[0][1:-1] for i in tmp_s_ann if i[5] == 'Gene']
        d_l = []
        try:
            d_l = [i[4].split('-')[0][1:-1] for i in tmp_s_ann if i[5] != 'Gene']
        except:
            print('warning', tmp_s_ann)
        d_l = [i[4].split('-')[0][1:-1] for i in tmp_s_ann if i[5] != 'Gene']
        if len(g_l) * len(d_l) > 0:
            all_p_l.extend([(i, j) for i in g_l for j in d_l])
            #print([(i, j) for i in g_l for j in d_l])
    tar_p_set = set(all_p_l)
    #print(len(tar_p_set))




    index = 0
    df_fixed_features = []
    df_word_sequences = []
    gdas = []
    for geneId in genes.keys():
        for diseaseId in diseases.keys():
            tmp_p = (geneId, diseaseId)
            if tmp_p not in tar_p_set:
                continue
            #print(tmp_p)
            find_target_gene = False
            find_target_disease = False
            target_gene = '<' + geneId + '>'
            target_disease = '<' + diseaseId + '>'
            pubmed_features = []
            for sentence_no in range(len(word_sequences)):
                sentence = word_sequences[sentence_no]
                features = fixed_features[sentence_no]
                new_features = []
                for token_no in range(len(sentence)):
                    feature = features[token_no]
                    ori_token = sentence[token_no]
                    token = ori_token
                    if isinstance(ori_token, tuple):
                        token = ori_token[0][0]
                        mes = ori_token[1][0]
                        #print(ori_token, token)

                    if target_disease in token.split('-') and target_gene in token.split('-'):
                        find_target_gene = True
                        find_target_disease = True
                        new_feature = [6]
                        new_features.append(new_feature)
                    elif (target_disease in token.split('-') and feature[0] == 1):
                        find_target_disease = True
                        new_feature = [4]
                        new_features.append(new_feature)
                    elif target_gene in token.split('-') and feature[1] == 1:
                        find_target_gene = True
                        new_feature = [5]
                        new_features.append(new_feature)
                    elif (target_disease not in token.split('-') and feature[0] == 1):
                        non_td = token.split('-')[0].strip('<>')
                        td = target_disease.strip('<>')
                        new_feature = [feature[0] + 2*feature[1]]
                        new_features.append(new_feature)
                    else:
#                         new_features.append([feature[0] + 2*feature[1]])
                        new_feature = [feature[0] + 2*feature[1]]
                        new_features.append(new_feature)
#                     if isinstance(ori_token, tuple) and new_features[-1][0] == 0:
#                         print(sentence, features)
                new_features = [_f[0] for _f in new_features]
                pubmed_features.append(new_features)
            if find_target_disease and find_target_gene:

                #_tmp_s, _tmp_f = [], []
                #for sentence_no in range(len(word_sequences)):
                #    tar_s = word_sequences[sentence_no][:]
                #    tar_f = pubmed_features[sentence_no]
                #    if ([4] in tar_f) or ([5] in tar_f) or ([6] in tar_f):
                #        _tmp_s.append(tar_s)
                #        _tmp_f.append(tar_f)
                #df_fixed_features.append(_tmp_f)
                #df_word_sequences.append(_tmp_s)

                df_fixed_features.append(pubmed_features)
                df_word_sequences.append(word_sequences)
                gdas.append([geneId, diseaseId])
            
        
    return df_word_sequences, df_fixed_features, gdas

def Generate_data_rnn(genes, diseases, word_sequences, fixed_features):
    index = 0
    df_fixed_features = []
    df_word_sequences = []
    gdas = []
    for geneId in genes.keys():
        for diseaseId in diseases.keys():
            find_target_gene = False
            find_target_disease = False
            target_gene = '<' + geneId + '>'
            target_disease = '<' + diseaseId + '>'
            pubmed_features = []
            for sentence_no in range(len(word_sequences)):
                sentence = word_sequences[sentence_no]
                features = fixed_features[sentence_no]
                new_features = []
                for token_no in range(len(sentence)):
                    feature = features[token_no]
                    ori_token = sentence[token_no]
                    token = ori_token
                    if isinstance(ori_token, tuple):
                        token = ori_token[0][0]
                        mes = ori_token[1][0]
                        #print(ori_token, token)

                    if target_disease in token.split('-') and target_gene in token.split('-'):
                        find_target_gene = True
                        find_target_disease = True
                        new_feature = [6]
                        new_features.append(new_feature)
                    elif (target_disease in token.split('-') and feature[0] == 1):
                        find_target_disease = True
                        new_feature = [4]
                        new_features.append(new_feature)
                    elif target_gene in token.split('-') and feature[1] == 1:
                        find_target_gene = True
                        new_feature = [5]
                        new_features.append(new_feature)
                    elif (target_disease not in token.split('-') and feature[0] == 1):
                        non_td = token.split('-')[0].strip('<>')
                        td = target_disease.strip('<>')
                        new_feature = [feature[0] + 2*feature[1]]
                        new_features.append(new_feature)
                    else:
#                         new_features.append([feature[0] + 2*feature[1]])
                        new_feature = [feature[0] + 2*feature[1]]
                        new_features.append(new_feature)
#                     if isinstance(ori_token, tuple) and new_features[-1][0] == 0:
#                         print(sentence, features)
                new_features = [_f[0] for _f in new_features]
                pubmed_features.append(new_features)
            if find_target_disease and find_target_gene:

                #_tmp_s, _tmp_f = [], []
                #for sentence_no in range(len(word_sequences)):
                #    tar_s = word_sequences[sentence_no][:]
                #    tar_f = pubmed_features[sentence_no]
                #    if ([4] in tar_f) or ([5] in tar_f) or ([6] in tar_f):
                #        _tmp_s.append(tar_s)
                #        _tmp_f.append(tar_f)
                #df_fixed_features.append(_tmp_f)
                #df_word_sequences.append(_tmp_s)

                df_fixed_features.append(pubmed_features)
                df_word_sequences.append(word_sequences)
                gdas.append([geneId, diseaseId])
            
        
    return df_word_sequences, df_fixed_features, gdas
