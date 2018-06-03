import xor_data
import train_mlp1
import mlp1
import numpy as np

dataList = []
for label, data in xor_data.data:
    dataList.append([label, np.array(data)])

params = mlp1.create_classifier(2, 2, 2)
trainedParams = train_mlp1.train_classifier(dataList, dataList, 800, 0.075, params)
