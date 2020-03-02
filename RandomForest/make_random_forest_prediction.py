import ezPickle as p
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
series_stats = pd.read_csv('data/summary_stats.csv')
test_stats = pd.read_csv('data/test_summary_stats.csv')
outputs = p.load('output_list')
for i in range(len(outputs)):
	outputs[i] = outputs[i].index(1)

print('load data done...')

clf = RandomForestClassifier(n_estimators=500, max_depth=7,random_state=0)

print('train ...')

clf.fit(series_stats.values, outputs)
p.save(clf,'rf_clf')

print('train done...')

predictions = clf.predict(test_stats.values)
def getReverseOneHotDict(dict):
    inverted_dict = {str(value): key for key, value in dict.items()}
    return inverted_dict

print('predict done...')

encoder_dict = p.load('encoder_dict')
decoder_dict = getReverseOneHotDict(encoder_dict)

predictions = predictions.tolist()

for i in range(len(predictions)):
	temp = [0]*9
	temp[predictions[i]] = 1	
	predictions[i] = temp
num_sequences = len(predictions)
output_data = pd.DataFrame({'series_id': range(num_sequences),'surface':[decoder_dict[str(item)] for item in predictions]})
output_data.to_csv('RF_Output.csv',index = False)
