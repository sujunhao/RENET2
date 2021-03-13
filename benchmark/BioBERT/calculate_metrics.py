import pandas as pd
import numpy as np
import subprocess

if __name__ == "__main__":
    
    # read gene/disease id
    gdas = []
    with open("./BeFree/n_out/cooc.befree") as f:
        for line in f:
            items = line.strip().split('\t')
            gdas.append([int(items[0].split('|')[0]),"F", items[9], items[9], int(items[7].split('|')[0]), items[10], items[11], items[13], items[16], items[17], items[19]])
    gdas_befree = pd.DataFrame(np.array(gdas), columns=["GAD_ID","GAD_ASSOC","GAD_GENE_SYMBOL","GAD_GENE_NAME","GAD_ENTREZ_ID","NER_GENE_ENTITY","NER_GENE_OFFSET","GAD_DISEASE_NAME","NER_DISEASE_ENTITY","NER_DISEASE_OFFSET","GAD_CONCLUSION"])
    gdas_befree.GAD_ID = gdas_befree.GAD_ID.astype(str)
    gdas_befree.GAD_ENTREZ_ID = gdas_befree.GAD_ENTREZ_ID.astype(str)

    gdas_for_biobert = gdas_befree  
    gdas_for_biobert = gdas_for_biobert[['GAD_ID', 'GAD_ENTREZ_ID', 'GAD_DISEASE_NAME']].copy()
    gdas_for_biobert = gdas_for_biobert.rename(columns={"GAD_ID": "pmid", "GAD_ENTREZ_ID": "geneId", \
                                                       "GAD_DISEASE_NAME": "diseaseId"})

    bert_f = './re_outputs_1/test_results.tsv'
    pred_df = pd.read_csv(bert_f, sep='\t', header=None)
    pred_df['pred'] = pred_df.apply(lambda x: 1 if x[1] > 0.5 else 0, axis = 1)

    gdas_for_biobert['pred'] = pred_df['pred']
    predicted_positive = gdas_for_biobert[gdas_for_biobert.pred==1][['pmid', 'geneId', 'diseaseId']].drop_duplicates()
    predicted_positive = predicted_positive.sort_values(['pmid', 'geneId', 'diseaseId'])
    predicted_positive.to_csv("./re_outputs_abs/classification_result_biobert.txt", index=False, columns=['pmid', 'geneId', 'diseaseId'])
    
    actual_positive = pd.read_csv('./BeFree/n_out/labels.txt')
    actual_positive = actual_positive[actual_positive['label'] == 1]

    #actual_positive = gdas_for_biobert[gdas_for_biobert['label'] == 1][['pmid', 'geneId', 'diseaseId']].drop_duplicates()
    print(actual_positive.shape)
    print(predicted_positive.shape)
    
    actual_positive.pmid = actual_positive.pmid.astype(str)
    actual_positive.geneId = actual_positive.geneId.astype(str)

    true_positives = pd.merge(actual_positive, predicted_positive, how="inner")

    precision = true_positives.shape[0] / float(predicted_positive.shape[0])

    recall = true_positives.shape[0] / float(actual_positive.shape[0])

    Fscore = 2 * precision * recall /(precision + recall)

    print("precision:{}  recall:{}  F-score:{}".format(precision, recall, Fscore))
    print("{},{},{}".format(precision, recall, Fscore))
