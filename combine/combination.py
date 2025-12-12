from itertools import groupby
import csv
from collections import Counter

import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, matthews_corrcoef
import numpy as np
from matplotlib import pyplot
import argparse
import time

def read_csv_2 (fname):

    label = []
    la = []
    pred = []
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        i = 0
        for line in reader:
            i += 1
            if i == 1:
                continue
            label.append(line[0])
            pred.append(float(line[1]))
            
              
    #print(len(pred), len(label))
    return pred, label



def eval_(y_true,y_pred, thresh=None):
    

    #print('size:', len(y_true), len(y_pred))
    auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    #auc_pc(y_true, y_pred)
    if thresh != None:
        y_pred = [ 1.0 if p> thresh else 0.0 for p in y_pred]
        
    
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    prc = precision_score(y_true=y_true, y_pred=y_pred)
    rc = recall_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    print('AUC:', auc)
    
    



## AUC-PC
# predict class values
def auc_pc(label, pred):
    lr_probs = np.array(pred)
    testy = np.array([float(l) for l in label])
    no_skill = len(testy[testy==1]) / len(testy)
    #yhat = np.array(pred)

    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
    #lr_f1 = f1_score(testy, yhat)
    #print(type(lr_precision), type(lr_recall))
    #print(np.shape(lr_precision), np.shape(lr_recall))
    lr_auc = auc(lr_recall, lr_precision)
    # summarize scores
    #print('AUC-PR:  auc=%.3f' % ( lr_auc))
    # plot the precision-recall curves

    pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    # pyplot.show()

    return lr_auc


parser = argparse.ArgumentParser()
parser.add_argument('-project', type=str, default='jdt')
parser.add_argument('-com_path', type=str, default='./com_pre/scl/')
parser.add_argument('-sim_path', type=str, default='./sim_pre/pred_scores/')
parser.add_argument('-sim_method', type=str, default='RandomForest')
parser.add_argument('-com_method', type=str, default='scl')
parser.add_argument('-seed', type=int, default=42)

args = parser.parse_args()


data_dir1 = "./com_pre/scl/"
data_dir2 = "./sim_pre/pred_scores/"

project = args.project

#Com
com_ = args.com_path

#Sim
sim_ = args.sim_path

## Simple add

flagSim = False
flagCom = False

try:
    pred, label = read_csv_2 (sim_)
    print("Prediction results of simple model was loaded !")
    flagSim = True
except:
    print("Didn't find prediction results of simple model, will try to load prediction results of complex model !")

try:
    pred_, label_ = read_csv_2 (com_)
    print("Prediction results of Complex model was loaded !")
    flagCom = True
except:
    if flagSim == True:
        print("Didn't find prediction results of complex model, only simple model results will be used !")
    else:
        print("Didn't find prediction results of both simple and complex model !")

t = 0.5
if flagSim == True and flagCom == True:
    pred2 = [pred_[i] + pred[i] for i in range(len(pred_))]
    t = 1
elif flagSim == True and flagCom == False:
    pred2 = pred
    label_ = label
elif flagSim == False and flagCom == True:
    pred2 = pred_


# pred2 = [ pred_[i] + pred[i] for i in range(len(pred_))]
#print(len(pred2), len(label_))
auc2 = roc_auc_score(y_true=np.array(label_),  y_score=np.array(pred2))
#print('\n SimCom: ')
mean_pred = float(sum(pred2)/len(pred2))
#eval_(y_true=np.array(label_),  y_pred=np.array(pred2), thresh = mean_pred )
pc_ = auc_pc(label_, pred2)


real_label = [float(l) for l in label_]
real_pred = [1 if p > t else 0 for p in pred2]
f1_ = f1_score(y_true=real_label, y_pred=real_pred)
recall = recall_score(real_label, real_pred, average='binary')
precision = precision_score(real_label, real_pred, average='binary')

# acc = accuracy_score(y_true=real_label, y_pred=real_pred)
mcc = matthews_corrcoef(real_label, real_pred)

print("AUC-ROC:{}  AUC-PR:{} percison:{} recall:{} F1-Score:{} MCC:{}".format(auc2, pc_, precision, recall, f1_, mcc))
# print("AUC-ROC:{}  AUC-PR:{} percison_2:{} recall_2:{} F1-Score_2:{} MCC:{}".format(auc2, pc_, precision_2, recall_2, f1_2, mcc))
print("done!")
df = pd.DataFrame({'label': label_, 'pred': pred2})
save_path = "./combination/{}/{}_{}_{}.csv".format(args.com_method,args.sim_method,args.seed,args.project)
df.to_csv(save_path, index=False, sep=',')
# df = pd.DataFrame({'project': [args.project], 'com_method': [args.com_method], 'sim_method': [args.sim_method], 'seed': [args.seed], 'AUC-ROC': [auc2],
#                    'AUC-PR': [pc_], 'precision': [precision], 'recall': [recall], 'F1-Score': [f1_],'MCC': [mcc]})
# df.to_csv('./all_ml_and_attention_msg_result.csv', index=False, sep=',', mode='a')