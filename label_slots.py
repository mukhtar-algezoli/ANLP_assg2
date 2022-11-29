import argparse
import json
import math
import os
import sklearn.linear_model
import spacy
import sys
import utils
from collections import defaultdict
import numpy as np

DATA_DIR = "data"
TRAINING_FILE = "training_data.json"
VALIDATION_FILE = "validation_data.json"

spacy.tokens.token.Token.set_extension("bio_slot_label", default="O")


def viterbi_algorithm(observations, states, start_p, trans_p, emit_p):
    V = [{}]
    print(observations)
    for st in states:
        #  print(observations[0])
        #  print(emit_p[st])
        #  print(emit_p[st][observations[0]])
         V[0][st] = {"prob": start_p[st] * emit_p[st][observations[0]], "prev": None}
   
    for t in range(1, len(observations)):
         V.append({})
         for st in states:
            max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
            prev_st_selected = states[0]
            for prev_st in states[1:]:
                tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
 
            max_prob = max_tr_prob * emit_p[st][observations[t]]
            V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
 
    opt = []
    max_prob = 0.0
    best_st = None
 
    for st, data in V[-1].items():
        if data["prob"] > max_prob:
            max_prob = data["prob"]
            best_st = st
    opt.append(best_st)
    previous = best_st
 
 
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    
    return opt


def _predict_tag_mle(token, model_parameters):
    """
    Predict most-frequent-tag baseline, i.e. argmax_{<tag>} P(<tag> | <word>).
    Args:
        token (spacy.tokens.Token): token whose tag we want to predict
        model_parameters (dict): a mapping token from most frequent tag, and an unknown word tag
    Returns:
        (List(Tuple(str,float))): a list of (tag, logprob) pairs, sorted from most to least probable
    """
    if token.text in model_parameters['per_word_tag_mle']:
        return model_parameters['per_word_tag_mle'][token.text]
    return model_parameters['unknown_word_distribution']


def train_tag_mle(data):
    """
    Train an unsmoothed MLE model P(<tag> | <word>).
    Args:
        data: data in NLU++ format imported from JSON, with spacy annotations.
    Returns:
        Callable: a function that returns a sorted distribution over tags, given a word.
    """
    token_count = 0

    word_tag_count = defaultdict(lambda: defaultdict(int))
    for sample in data:
        for token in sample['annotated_text']:
            word_tag_count[token.text][token._.bio_slot_label] += 1
            token_count += 1

    per_word_tag_mle = {}
    for word, tag_count_pairs in word_tag_count.items():
        word_count = sum(tag_count_pairs.values())
        tag_logprob_pairs = [(tag, math.log(tag_word_count / word_count))
                             for (tag, tag_word_count) in tag_count_pairs.items()]
        sorted_tag_logprob_pairs = sorted(
            tag_logprob_pairs, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
        per_word_tag_mle[word] = sorted_tag_logprob_pairs

    model_parameters = {
        'per_word_tag_mle': per_word_tag_mle,
        # Assign O for unknown words with prob 1
        # but since we are using log probs -> log(1) = 0.
        'unknown_word_distribution': [('O', 0.0)]
    }

    print(
        f'> Finished training most frequent tag model on {token_count} tokens')
    return lambda token: _predict_tag_mle(token, model_parameters)


def _predict_tag_logreg(token, prev_token, model, tags):
    """
    Predict tag according to logistic regression on word embedding
    Args:
        data: data in NLU++ format impored from JSON, with spacy annotations.

    Returns:
        (List(Tuple(str,float))): distribution over tags, sorted from most to least probable
    """
    # if prev_token:
    #     print("prev token is: " + str(prev_token.text))
    #     print("curr token is: " + str(token.text))
    #     print("|||||||||||||||||||||||||||||||||||||||||||||||")
    vec = token.vector
    vec = np.append(vec, token.is_alpha)
    vec = np.append(vec, token.is_stop * 10)
    vec = np.append(vec, token.dep/2**64)
    vec = np.append(vec, token.is_digit * 10)
    if prev_token:
        vec = np.append(vec, prev_token.vector)
    else:
        # vec = np.append(vec, [0] * len(token.vector))
        vec = np.append(vec, [0] * len(token.vector))
    # if token.is_digit:
    #     print(token.is_digit)
    log_probs = model.predict_log_proba(
        [vec])[0]
    distribution = [
        tag_logprob_pair for tag_logprob_pair in zip(tags, log_probs)]
    sorted_distribution = sorted(
        distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution


def train_tag_logreg(data):
    """
    Train a logistic regression model for p(<tag> | <word embedding>).

    Args:
        data: data in NLU++ format imported from JSON, with spacy annotations.
    Returns:
        Callable: a function that returns a (sorted) distribution over tags, given a word.
    """
    #counts the tokens
    token_count = 0
    # training input features (embeddings in logistic regression)
    train_X = []
    # training targets 
    train_y = []
    # set that include biotags
    bio_tags = set(['O'])
    # this loop iterate over the anotated training samples 
    #and collecte all possible bio_slot_labels
    for sample in data: # iterating over the anotated sentences
        for token in sample['annotated_text']: # get the text for each input 
            bio_tags.add(token._.bio_slot_label) #adds the biotag label to bio_tags

    bio_tags = sorted(bio_tags)
    # maps each tag to a unique index to make it appropriate as 
    #a label to logistic regression (not necessary)
    bio_tag_map = {tag: idx for idx, tag in enumerate(bio_tags)}

    #iterate over each anotated sample in the training data
    for sample in data:
        # this variable is used to save the previous token so that we can use it as
        #extra embedding to the logistic regression model
        prev_token = None
        # iterate over the tokens for each example
        for i, token in enumerate(sample["annotated_text"]):
            # count the number of tokens in our data
            token_count += 1
            # invoke the embeddings from Spacy annotation
            vec = token.vector
            # append the is_alpha value to the end of our vector
            #this vector tells us if the vector has chars or not
            vec = np.append(vec, token.is_alpha)
            # append is_stop value to the end of our vector
            #is_stop tells us if the vector is_stop word or not (True, False)
            # we extend the is_stop array to length of ten to emphasize its 
            # importance 
            vec = np.append(vec, token.is_stop * 10)
            # append dep value to the end of our vector
            #dep value gives the Syntactic dependency relation.
            vec = np.append(vec, token.dep/2**64)
            # append the is_digit value to the end of our vector
            #this vector tells us if the vector is a number or not
            # we extend the is_digit array to length of ten to emphasize its 
            # importance 
            vec = np.append(vec, token.is_digit * 10)
            # for each token if it is not in the beginning of the sentence 
            # (has previous vector) we append that previous vector 
            # embeddings to the end of our vector if our word in the beginning
            #  of the sentence we append zeros
            if prev_token:
                vec = np.append(vec, prev_token.vector)
            else:
                vec = np.append(vec, [0] * len(token.vector))
            # append the embeddings vector of the current token to the input features
            train_X.append(vec)
            # append the BIOtag to the target labels list
            train_y.append(bio_tag_map[token._.bio_slot_label])
            # current token to be used as previous token to the next token in
            #the sentence
            prev_token = token

    print(f'> Training logistic regression model on {token_count} tokens')
    model = sklearn.linear_model.LogisticRegression(
        multi_class='multinomial', solver='newton-cg').fit(train_X, train_y)
    print('> Finished training logistic regression')
    return lambda token, prev_token: _predict_tag_logreg(token, prev_token, model, bio_tags)


def predict_independent_tags(tag_predictor, validation_data, training_data):
    '''
    Predict tags independently of each other.

    Args:
        tag_predictor: function that predicts a tag, given a spacy token
        validation_data: validation_data in NLU++ format imported from JSON, with spacy annotations
    Returns:
        List(List(Str)): a list (one for each sample) of a list of tags (one for each word in the sample)
    '''
    predictions = []
    for sample in validation_data:
        predictions.append([tag_predictor(token, "prev tag")[0][0]
                           for token in sample['annotated_text']])
    return predictions


def predict_bio_tags(tag_predictor, validation_data, 
                training_data):
    '''
    Predict tags respecting BIO tagging constraints.
    
    Args:
        tag_predictor: function that predicts a tag, given a spacy token
        data: data in NLU++ format imported from JSON, with spacy annotations

    Returns:
        List(List(Str)): a list (one for each sample) of a list of tags (one for each word in the sample)

    '''
    # we added the training data to function arguments
    #to calculate bigrams sequences from it

    # variable control which method we use,
    #either greedy or viterbi
    method = "greedy"
    # this list variable will contains lists each 
    #list represent the biotags of a sentence in
    #the training data 
    samples_biotags = []
    
    bio_tags = set(['O'])
    # this loop iterate over the anotated samples
    # in the validation data and collecte all 
    # possible bio_slot_labels
    for sample in validation_data:# iterating over the anotated sentences
        for token in sample['annotated_text']:# get the text for each input
            bio_tags.add(token._.bio_slot_label)#adds the biotag label to bio_tags
            # print(token._.bio_slot_label)


    # iterate over annotated sentences in the training data
    #building samples_biotags
    for sample in training_data:
        samples_biotags_example = []
        for token in sample["annotated_text"]:
            samples_biotags_example.append(token._.bio_slot_label)
        samples_biotags.append(samples_biotags_example)

    # this list contains all possible bio tags
    the_bio_tags = ['B-adults', 'I-adults', 'B-rooms', 
    'I-rooms', 'O', 'I-date', 'B-kids', 'B-date_from', 
    'I-people', 'B-date_to', 'I-time_to', 'B-time_period', 
    'I-date_to',  'I-date_from', 'B-person_name', 'I-time', 
    'B-time', 'B-number', 'I-kids',  'B-time_to', 
    'B-time_from', 'I-person_name', 'I-time_from', 
    'I-date_period', 'I-time_period', 'B-date_period', 
    'B-date', 'B-people']

    # This nested dictionary will contain the probabilities for each
    #bigram tags (0 means it is not valid)
    tags_bigrams_dict = defaultdict(dict)
    for i in range(len(the_bio_tags)):#iterate over tags
        for j in range(len(the_bio_tags)): #iterate over tags
            counter = 0 #counts the number of time a tag bigram appear
            for sample in samples_biotags: # iterate over the sample sentences
                for k in range(len(sample)):# iterate over the tokens in each sample
                    # if the the current token and the next one is equal
                    #to the two BIOtags we are looking in, increment the counter
                    #for that Bigram Bio tag
                    if sample[k] == the_bio_tags[i] and k+1 <= len(sample) - 1 and sample[k + 1] == the_bio_tags[j]:
                        counter += 1
            # is a biotags bigram apear more than once set it as valid
            # valid == 1, invalid == 0
            tags_bigrams_dict[the_bio_tags[i]][the_bio_tags[j]] = 1 if counter > 0  else 0
    
    #set the probability for each tag bigram by counting the number of
    #valid bigrams with the the first tag and then dividing by that number
    #each bigram with that first tag
    for first_tag in tags_bigrams_dict:
        tag_count = sum(tags_bigrams_dict[first_tag].values())
        for second_tag in tags_bigrams_dict[first_tag]:
            tags_bigrams_dict[first_tag][second_tag] = tags_bigrams_dict[first_tag][second_tag]/tag_count 
    

    #this will contain our output predictions
    predictions = []

    #Viterbi implementation
    if method == "viterbi":
        #this will be our emission matrix 
        #a nested dict with tag as first key
        #word as second key and probability
        #as the value
        emission_mat = defaultdict(dict)

        # Iterate over each token in the validation data and
        #run it through the model and append each (predicted tag, text of token,
        # predicted probability) to emission_mat
        for sample in validation_data: #iterating over samples
            for index, token in enumerate(sample['annotated_text']): #iterating over tokens
                for i in range(len(tag_predictor(token))): #iterating over sorted predictions
                    #tag_predictor(token) produces the logistic regression 
                    #predictions
                    emission_mat[tag_predictor(token)[i][0]][token.text] = math.exp(tag_predictor(token)[i][1])
                    #we apply exponent to the prediciton because the model
                    #produces the log of the probability
        #Viterbi states is the list of biotags
        states = the_bio_tags
        #the probability to start with each biotag
        # it is 1/14 because all "I" tags can't be
        #in the beginning of the list 
        start_p = {}
        for stat in states:
            if stat[0] != "I":
                start_p[stat] = 1/14
            else:
                start_p[stat] = 0 
        
        
        # iterate over the sentences and apply the Viterbi algorithm to each
        #anotated sentence in the validation data
        for sample in validation_data: #iterating over samples
            sentence_list = [token.text for token in sample['annotated_text']]
            #tags_bigrams_dict == transtions matrix
            predictions.append(viterbi_algorithm(sentence_list, states, start_p, tags_bigrams_dict, emission_mat))

    # the method value is greedy we use greedy algorithm instead
    elif method == "greedy":
        #iterate over the samples in the validaiton_data
        for sample in validation_data: #iterating over samples
            #we use prev prediction to check if a biotag and
            #the one before it is valid or not
            prev_prediction = " "
            #predictions for each annotated sentence
            sample_predictions = []
            
            #iterate over the tokens in each sample
            for index, token in enumerate(sample['annotated_text']): #iterating over tokens
                #iterating over sorted predictions for that token
                #tag_predictor(token) produces the logistic regression 
                #predictions
                for i in range(len(tag_predictor(token))): 
                    # if the index in the beginning of the 
                    #sentence then accept the first prediction
                    if index <= 0:
                        prev_prediction = tag_predictor(token)[i][0]
                        sample_predictions.append(tag_predictor(token)[i][0])
                        break
                    else:
                        #if prev_tag, current prediction is not valid
                        #its probability is zero continue to next prediction
                        #for that token, else put it as the output
                        if tags_bigrams_dict[prev_prediction][tag_predictor(token)[i][0]] > 0.0:
                            prev_prediction = tag_predictor(token)[i][0]
                            sample_predictions.append(tag_predictor(token)[i][0])
                            break
                        else:
                            continue
            #append the output predictions for the sentence                            
            predictions.append(sample_predictions)

    ###
    ### Include the code in Appendix B of your report
    ################################################################
   
    return predictions


train = {
    'most_frequent_tag': train_tag_mle,
    'logistic_regression': train_tag_logreg
}
default_train = 'most_frequent_tag'

predict = {
    'independent_tags': predict_independent_tags,
    'bio_tags': predict_bio_tags,
}
default_predict = 'independent_tags'

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--train_path',
                   help='path to training dataset in JSON format',
                   default=os.path.join(DATA_DIR, TRAINING_FILE))
    p.add_argument('-v', '--validation_path',
                   help='path to validation dataset in JSON format',
                   default=os.path.join(DATA_DIR, VALIDATION_FILE))
    p.add_argument('-m', '--model',
                   help=f'type of model to use.',
                   choices=train.keys(),
                   default=default_train)
    p.add_argument('-p', '--predictor',
                   help=f'type of prediction method to use.',
                   choices=predict.keys(),
                   default=default_predict)
    p.add_argument('-f', '--full_report',
                   help=f'show complete output of tagging model on all sentences',
                   action='store_true')
    args = p.parse_args()

    with open(args.train_path) as f:
        training_data = json.load(f)
    with open(args.validation_path) as f:
        validation_data = json.load(f)

    print("> Tokenising and annotating raw data")
    nlp_analyser = spacy.load("en_core_web_sm")
    # nlp_analyser = spacy.load("en_core_web_trf")
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(
        training_data, nlp_analyser)
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(
        validation_data, nlp_analyser)

    print(f'> Training model on ')
    model = train[args.model](training_data)

    print(f'> Predicting tags on validation set')
    predictions = predict[args.predictor](
        model, validation_data, training_data)

    if args.full_report:
        utils.visualise_bio_tags(predictions, validation_data)

    utils.evaluate(predictions, validation_data)
