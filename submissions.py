from keras.models import Model
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from preprocess import convnet
import csv

file = pd.read_csv('test.csv')
sequence = file['sequence']
alphabet = 'ACGT'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
N = len(sequence)
integer_encoded = np.zeros((N,14))
for i in range(len(sequence)):
    integer_encoded[i] = [char_to_int[char] for char in sequence[i]]
encoded = to_categorical(integer_encoded)
alphabetToInteger = np.transpose(encoded, (0, 2, 1))
alphabetToIntegerTest = alphabetToInteger[..., np.newaxis]

model =  convnet(input_shape=(4,14,1))
model.load_weights('weight.h5')
predictedlabels = model.predict(alphabetToIntegerTest, verbose=1, batch_size=128)

accuracyLabels = np.zeros(predictedlabels.shape[0])

i=0
for pred in predictedlabels:
    if pred[0]>=0.5:
        accuracyLabels[i] = 0
    else:
        accuracyLabels[i] = 1
    i+=1
for i,p in enumerate(accuracyLabels):
  accuracyLabels[i] = int(p)
  
  
N = np.shape(accuracyLabels)
print (accuracyLabels[0])

with open('submission.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "prediction"])
    for i in range(400):
        myfile.write("{}, {}\n".format(i, int(accuracyLabels[i])))
        
myfile.close()   
