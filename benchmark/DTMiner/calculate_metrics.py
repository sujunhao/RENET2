import pandas as pd
import subprocess

if __name__ == "__main__":

    candidate_gdas = pd.read_csv("n_out/candidate_GDAs.csv", names=['pmid', 'diseaseId', 'geneId'], sep="\t")
    
    predict = pd.read_csv("n_out/DTMiner_predict", names=['predict'])
    
    candidate_gdas['predict'] = predict.predict
    
    predicted_positive = candidate_gdas[candidate_gdas.predict==1][['pmid', 'geneId', 'diseaseId']].drop_duplicates()
    predicted_positive.to_csv("n_out/classification_result_dtminer.txt", index=False)
    
    
    actual_positive = pd.read_csv('n_out/labels.txt')
    actual_positive = actual_positive[actual_positive['label'] == 1]
    #actual_positive = pd.read_csv('/mnt/bal31/jhsu/old/data/new_renet_data/test/labels.txt')
    print(actual_positive.shape)
    print(predicted_positive.shape)

    actual_positive.pmid = actual_positive.pmid.astype(str)                                                                                                 
    predicted_positive.pmid = predicted_positive.pmid.astype(str)                                                                                                 
    predicted_positive.geneId = predicted_positive.geneId.astype(str)                                                                                                 
    actual_positive.geneId = actual_positive.geneId.astype(str)

    
    true_positives = pd.merge(actual_positive, predicted_positive, how="inner")
    
    precision = true_positives.shape[0] / float(predicted_positive.shape[0])
    
    recall = true_positives.shape[0] / float(actual_positive.shape[0])
    
    Fscore = 2 * precision * recall /(precision + recall)
    
    print("precision:{}  recall:{}  F-score:{}".format(precision, recall, Fscore))
    print("{},{},{}".format(precision, recall, Fscore))
    
    #subprocess.call(["rm", "out/feature.hybrid"])
    #subprocess.call(["rm", "out/DTMiner_predict"])
    #subprocess.call(["rm", "out/candidate_GDAs.csv"])
