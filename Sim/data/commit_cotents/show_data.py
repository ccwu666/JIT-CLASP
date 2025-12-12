#%%
import pickle
from collections import Counter

f = 'C:\\Users\\Hugo\\Desktop\\DP\\SimCom_JIT\\data\\commit_cotents\\processed_data\\openstack\\openstack_train.pkl'

data = pickle.load(open(f, 'rb'))
#print(data_path+file_name)
print(f)
print(len(data[0]))
ids, labels, msgs, codes = data
print(len(ids))

print(ids[12])
print(labels[12])
print(codes[12])
print(len(codes[12]))
print(msgs[12])

