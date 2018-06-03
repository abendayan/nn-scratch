import mlp1 as ml1
import numpy as np
import random
import utils as ut
import mlp1 as ml

def feats_to_vec(features):
    # Should return a numpy vector of features.
    return np.array(features)

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        if ml1.predict(features, params) == label:
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
    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = label                 # convert the label to number if needed.
            loss, grads = ml1.loss_and_gradients(x,y,params)
            cum_loss += loss

            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0][0] -= grads[0][0]*learning_rate
            params[0][1] -= grads[0][1]*learning_rate
            params[1][0] -= grads[1][0]*learning_rate
            params[1][1] -= grads[1][1]*learning_rate

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def build_data(data, bigrams, lang):
    return_data = []
    for [language, bs] in data:
        features = np.zeros(len(bigrams))
        for b in bs:
            if b in bigrams:
                features[bigrams[b]] += 1
        final_lang = lang[language] if language in lang else -1
        return_data.append([final_lang, features])
    return return_data


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    # ...
    bigrams = {}
    lang = {}
    lang_previous = {}

    i = 0
    j = 0
    for [language, text] in ut.TRAIN:
        if language not in lang:
            lang[language] = i
            lang_previous[i] = language
            i += 1
        for element in text:
            if element not in bigrams:
                bigrams[element] = j
                j += 1
    in_dim = len(bigrams)
    out_dim = len(lang)
    train_data = build_data(ut.TRAIN, bigrams, lang)
    dev_data = build_data(ut.DEV, bigrams, lang)
    test_data = build_data(ut.TEST, bigrams, lang)
    num_iterations = 500
    learning_rate = 0.0001
    hid_dim = int(np.log(in_dim * out_dim))
    params = ml1.create_classifier(in_dim, hid_dim, out_dim)
    trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)

    results = []
    for [label, data] in test_data:
        results.append(lang_previous[ml.predict(data, trained_params)])

    output_file = open('test.pred', 'w')
    output_file.write(ut.list_to_string(results))
    output_file.close()
