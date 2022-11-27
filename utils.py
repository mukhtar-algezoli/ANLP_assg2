import itertools
from collections import defaultdict
from sklearn import metrics

def tokenise_annotate_and_convert_slot_labels_to_bio_tags(data, nlp_analyser):
   """
   Preprocesses the raw data:
    - tokenises and analyse using spacy, storing result under the 'analysis' key 
      as a spacy Doc object
    - convert span annotations on slots into IOB tags on tokens, storing tag 
      under 'bio_slot_label' for each spacy Token object
   Data is modified in place 

   Args:
       data (dict): raw data 
       nlp_analyser (spacy.lang.<Language>): spacy model
   """ 
#    print(nlp_analyser.components)
   for sample in data:
       if 'text' in sample:
           sample['annotated_text'] = nlp_analyser(sample['text'])
        #    for token in sample['annotated_text']:
                # print("///////////////////////////")
                # # print(token.keys)
                # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                #     token.shape_, token.is_alpha, token.is_stop)
       if 'slots' in sample:
           for slot in sample['slots']:
               if 'span' in sample['slots'][slot]:
                   (span_start, span_end) = sample['slots'][slot]['span']
                   found_first_token = False
                   for token in sample['annotated_text']:
                       if token.idx < span_start:
                           continue
                       if token.idx >= span_end:
                           break
                       if found_first_token:
                           token._.bio_slot_label= f'I-{slot}'
                       else: 
                           token._.bio_slot_label= f'B-{slot}'
                           found_first_token = True

def _extract_spans(tag_sequence):
    ''''
    Return a list of labeled spans from a tagged sequence
    
    Args:
        List(str): a list of tags in BIO format
    Returns:
        Bool: True if input BIO tag sequence is well-formed, False otherwise
        List(Tuple(str,(int,int))): a list of labels and associated spans
    '''
    spans = defaultdict(lambda: '<not_a_span>') 
    in_label = False
    wellformed_bio_tags = True
    for idx, tag in enumerate(tag_sequence + ['O']):
        if tag.startswith('B'):
            if in_label:
               spans[(span_start, idx)] = label
            in_label = True
            label = tag[2:]
            span_start = idx  
        elif tag.startswith('I'):
            if not (in_label and label == tag[2:]):
                wellformed_bio_tags = False
        elif tag == 'O':
            if in_label:
               spans[(span_start, idx)] = label
               in_label = False 
        else:
            wellformed_bio_tags = False 
    return wellformed_bio_tags, spans

def evaluate(predictions, validation_data):
    '''
    Evaluate predicted BIO tags and slot labels, and print report

    Args:
        predictions (List(List(str))): predictions for each sample, each consisting of one tag per word
        validation_data: data in NLU++ format imported from JSON, with true spans and corresponding BIO tags
    '''
    correct_tags = []
    predicted_tags = [] 
    
    correct_spans = []
    predicted_spans = []
    span_labels = set()

    corpus_level_well_formed_bio_tags = True
    for sample, prediction in zip(validation_data, predictions):
        sample_tags =  [token._.bio_slot_label for token in sample['annotated_text']]
        correct_tags.extend(sample_tags)
        predicted_tags.extend(prediction)
        
        well_formed_bio_tags, sample_spans = _extract_spans(sample_tags)
        assert well_formed_bio_tags, "Annotated BIO tags are malformed. This shouldn\'t happen!"
        well_formed_bio_tags, prediction_spans = _extract_spans(prediction)
        corpus_level_well_formed_bio_tags = corpus_level_well_formed_bio_tags and well_formed_bio_tags
        
        labeled_spans = set(itertools.chain(sample_spans.keys(), prediction_spans.keys()))
        for span in labeled_spans:
            correct_spans.append(sample_spans[span])
            predicted_spans.append(prediction_spans[span])
            span_labels.add(sample_spans[span])
            span_labels.add(prediction_spans[span])

    span_labels -= set(['<not_a_span>'])
    span_labels = sorted(span_labels)
    tag_labels = sorted([f'B-{label}' for label in span_labels] + [f'I-{label}' for label in span_labels])

    print()
    print("Classification report for BIO tags:")
    print()
    report = metrics.classification_report(correct_tags, predicted_tags, labels=tag_labels, zero_division=0)
    print(report)

    if corpus_level_well_formed_bio_tags:
        print()
        print("Classification report for slot labels:")
        print()
        report = metrics.classification_report(correct_spans, predicted_spans, labels=span_labels, zero_division=0)
        print(report)
        print()
    else:
        print('!!! Some BIO tag sequences were malformed, so spans may not have been correctly extracted from them !!!')

def visualise_bio_tags(predictions, validation_data):
    '''
    Print out predicted BIO tags and slot labels vs true tags and labels, and print report

    Args:
        predictions (List(List(str))): predictions for each sample, each consisting of one tag per word
        validation_data: data in NLU++ format imported from JSON, with true spans and corresponding BIO tags
    '''
    longest_tok_len = 0
    for sample, prediction, in zip(validation_data, predictions):
        print(f"Input: {sample['text']}")
        print()
        print("Analysis:                                                     Additional annotations:                           ")
        print("idx word            true label      predicted label correct?  lemma           tag      dep      shape           ")
        print("------------------------------------------------------------  --------------------------------------------------")
        for idx, (token, predicted_label) in enumerate(zip(sample['annotated_text'], prediction)):
            match = '  ✔     ' if token._.bio_slot_label == predicted_label else '     ✗  '
            print(f'{idx:<3} {str(token):<15} {token._.bio_slot_label:<15} {predicted_label:<15} {match:<9} {token.lemma_:<15} {token.tag_:<8} {token.dep_:<8} {token.shape_:<15}')
            longest_tok_len = max(longest_tok_len, len(token))
        print()
        well_formed_bio_tags, annotated_spans = _extract_spans([token._.bio_slot_label for token in sample['annotated_text']])
        assert well_formed_bio_tags, "Annotated BIO tags are malformed. This shouldn\'t happen!"
        well_formed_bio_tags, predicted_spans = _extract_spans(prediction)
        if not well_formed_bio_tags:
            print('!!! Predicted BIO tag sequence is malformed. Some predicted spans were not extracted correctly !!!')
            print()
        print("true spans:")
        for span, label in annotated_spans.items():
            print(f'  {span}: {label}')
        print()
        print("predicted spans:")
        for span, label in predicted_spans.items():
            print(f'  {span}: {label}')
        print()
        print("================================================================================================================")
        print()
 
