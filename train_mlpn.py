import mlpn
import numpy as np
import random
from math import log

def feats_to_vec(features):
    return features


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        if mlpn.predict(features, params) == label:
            good += 1
        else:
            bad += 1
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = label  # convert the label to number if needed.
            loss, grads = mlpn.loss_and_gradients(x, y, params)
            cum_loss += loss

            for i in range(len(params)):
                params[i][0] -= grads[i][0] * learning_rate
                params[i][1] -= grads[i][1] * learning_rate

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    i = 0
    bigrams = {}
    j = 0
    languages = {}
    languagesBack = {}

    import utils

    for [lang, bs] in utils.TRAIN:
        if lang not in languages:
            languages[lang] = j
            languagesBack[j] = lang
            j += 1
        for b in bs:
            if b not in bigrams:
                bigrams[b] = i
                i += 1


    def dataFromFile(fileData):
        data = []
        for [lang, bs] in fileData:
            features = np.zeros(len(bigrams))
            for b in bs:
                if b in bigrams:
                    features[bigrams[b]] += 1
            language = languages[lang] if lang in languages else -1
            data.append([language, features])
        return data


    trainData = dataFromFile(utils.TRAIN)
    devData = dataFromFile(utils.DEV)
    testData = dataFromFile(utils.TEST)

    params = mlpn.create_classifier([len(bigrams), 13, 13, len(languages)])
    trainedParams = train_classifier(trainData, devData, 20, 0.01, params)

    predictions = []
    for [label, data] in testData:
        predictions.append(languagesBack[mlpn.predict(data, trainedParams)])

    outF = open('test.pred', 'w')
    outF.write("\n".join(predictions))
    outF.close()
