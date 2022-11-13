import argparse
import json
import math
import os
import sklearn.linear_model
import spacy
import sys
import utils
from collections import defaultdict

DATA_DIR = "data"
TRAINING_FILE = "training_data.json"
VALIDATION_FILE = "validation_data.json"

spacy.tokens.token.Token.set_extension("bio_slot_label", default="O")

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
        tag_logprob_pairs = [(tag, math.log(tag_word_count / word_count)) for (tag, tag_word_count) in tag_count_pairs.items()]
        sorted_tag_logprob_pairs = sorted(tag_logprob_pairs, key=lambda tag_logprob_pair: -tag_logprob_pair[1]) 
        per_word_tag_mle[word] = sorted_tag_logprob_pairs

    model_parameters = {
        'per_word_tag_mle': per_word_tag_mle,
        # Assign O for unknown words with prob 1
        # but since we are using log probs -> log(1) = 0.
        'unknown_word_distribution': [('O', 0.0)]
    }

    print(f'> Finished training most frequent tag model on {token_count} tokens')
    return lambda token: _predict_tag_mle(token, model_parameters)

def _predict_tag_logreg(token, model, tags):
    """
    Predict tag according to logistic regression on word embedding

    Args:
        data: data in NLU++ format impored from JSON, with spacy annotations.
    
    Returns:
        (List(Tuple(str,float))): distribution over tags, sorted from most to least probable
    """
    log_probs = model.predict_log_proba([token.vector])[0]
    distribution = [tag_logprob_pair for tag_logprob_pair in zip(tags, log_probs)]
    sorted_distribution = sorted(distribution, key=lambda tag_logprob_pair: -tag_logprob_pair[1])
    return sorted_distribution

def train_tag_logreg(data):
    """
    Train a logistic regression model for p(<tag> | <word embedding>).
    
    Args:
        data: data in NLU++ format imported from JSON, with spacy annotations.

    Returns:
        Callable: a function that returns a (sorted) distribution over tags, given a word.
    """
    token_count = 0
    train_X = [] 
    train_y = []
    bio_tags = set(['O'])
    for sample in data:
        for token in sample['annotated_text']:
            bio_tags.add(token._.bio_slot_label)

    bio_tags = sorted(bio_tags)
    bio_tag_map = {tag: idx for idx, tag in enumerate(bio_tags)}

    ################################################################
    ### TODO: Write code to set train_X, train_y, and token_count
    for sample in data:
        for token in sample["annotated_text"]:
            train_X.append(token.vector)
            train_y.append(bio_tag_map[token._.bio_slot_label])

    ### Include the code in Appendix A of your report
    ################################################################

    print(f'> Training logistic regression model on {token_count} tokens')
    model = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_X, train_y)
    print('> Finished training logistic regression')
    return lambda token: _predict_tag_logreg(token, model, bio_tags)


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
        predictions.append([tag_predictor(token)[0][0] for token in sample['annotated_text']])
    return predictions

def predict_bio_tags(tag_predictor, validation_data, training_data):
    '''
    Predict tags respecting BIO tagging constraints.
    
    Args:
        tag_predictor: function that predicts a tag, given a spacy token
        data: data in NLU++ format imported from JSON, with spacy annotations

    Returns:
        List(List(Str)): a list (one for each sample) of a list of tags (one for each word in the sample)

    '''
    samples_biotags = []
    bio_tags = set(['O'])
    for sample in validation_data:
        for token in sample['annotated_text']:
            bio_tags.add(token._.bio_slot_label)
            # print(token._.bio_slot_label)

    bio_tags = sorted(bio_tags)



    ################################################################
    ### TODO: Write code to predict tags, obeying BIO constraints

    for sample in training_data:
        samples_biotags_example = []
        for token in sample["annotated_text"]:
            samples_biotags_example.append(token._.bio_slot_label)
        samples_biotags.append(samples_biotags_example)

    the_bio_tags = ["B-adults", "I-adults", "B-rooms", "I-rooms", "O"]
    the_bio_tags = ['B-adults', 'I-adults', 'B-rooms', 'I-rooms', 'O', 'I-date', 'B-kids', 'B-date_from', 'I-people', 'B-date_to', 'I-time_to', 'B-time_period', 'I-date_to',  'I-date_from', 'B-person_name', 'I-time', 'B-time', 'B-number', 'I-kids',  'B-time_to', 'B-time_from', 'I-person_name', 'I-time_from', 'I-date_period', 'I-time_period', 'B-date_period', 'B-date', 'B-people']
    tags_bigrams = []

    for i in range(len(the_bio_tags)):
        for j in range(len(the_bio_tags)):
            counter = 0
            for sample in samples_biotags:
                for k in range(len(sample)):
                    if sample[k] == the_bio_tags[i] and k+1 <= len(sample) - 1 and sample[k + 1] == the_bio_tags[j]:
                        counter += 1
            tags_bigrams.append({"first_biotag":the_bio_tags[i], "second_biotag":the_bio_tags[j], "is_valid":1 if counter > 0  else 0})


    predictions = []
    for sample in validation_data: #iterating over samples
        prev_prediction = " "
        sample_predictions = []
        
        for index, token in enumerate(sample['annotated_text']): #iterating over tokens
            for i in range(len(tag_predictor(token))): #iterating over sorted predictions
                if index <= 0:
                    prev_prediction = tag_predictor(token)[i][0]
                    sample_predictions.append(tag_predictor(token)[i][0])
                    break
                else:
                    if [tag for tag in tags_bigrams if tag["first_biotag"] == prev_prediction and tag["second_biotag"] == tag_predictor(token)[i][0]][0]["is_valid"]:
                        prev_prediction = tag_predictor(token)[i][0]
                        sample_predictions.append(tag_predictor(token)[i][0])
                        break
                    else:
                        continue
                        
        predictions.append(sample_predictions)

    ###
    ### Include the code in Appendix B of your report
    ################################################################
   
    return predictions

  
train = {
    'most_frequent_tag' : train_tag_mle,
    'logistic_regression' : train_tag_logreg
}
default_train = 'most_frequent_tag'

predict = {
    'independent_tags' : predict_independent_tags,
    'bio_tags' : predict_bio_tags,
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
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(training_data, nlp_analyser)
    utils.tokenise_annotate_and_convert_slot_labels_to_bio_tags(validation_data, nlp_analyser)

    print(f'> Training model on ')
    model = train[args.model](training_data)

    print(f'> Predicting tags on validation set')
    predictions = predict[args.predictor](model, validation_data, training_data)
    
    if args.full_report:
        utils.visualise_bio_tags(predictions, validation_data)

    utils.evaluate(predictions, validation_data)
