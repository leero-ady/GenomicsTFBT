import pandas as pd
import csv


file = pd.read_csv('submission.csv')
prediction = file['prediction']
id=file['id']
with open('submissions.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    writer.writerow(["id", "prediction"])
    for id, prediction in zip(id, prediction):
        myfile.write("{},{}\n".format(id,int(prediction)))

myfile.close()