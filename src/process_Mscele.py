import base64
import csv
import os


file = '/dataset/MsCelebV1-Faces-Aligned.tsv'

with open(file,'r') as tsvF:
    reader = csv.reader(tsvF, delimiter='\t')
    i = 0
    for row in reader:
        MID, imgSearchRank, faceID, data = row[0], row[1], row[4], base64.b64decode(row[-1])
        saveDir = os.path.join('/dataset/MsCele_aligned', MID)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir) 
        savePath = os.path.join(saveDir, "{}-{}.jpg".format(imgSearchRank, faceID))
        with open(savePath, 'wb') as f:
            f.write(data)
        i+=1
        if i%10000 == 0:
            print i
    print i
