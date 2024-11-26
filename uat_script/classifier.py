import os
from tqdm.auto import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from torch import no_grad, tensor
import json

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.join(os.path.dirname(__file__),'../'))
# global config
config = load_config(proj_home=proj_home)
logger = setup_logging('classifier.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))


def load_uat():

    with open(config['UAT_JSON_PATH'], 'r') as f:
        uat_list = json.load(f)

# build the dict that matches UAT ID (numbers) to common names
    uat_names = {}
    for entry in uat_list:
        uat_id = int(entry['uri'].split('/')[-1])
        uat_names[uat_id] = entry['name'].lower().strip()
        # dont add the alt names
        # if 'altNames' in entry.keys() and entry['altNames'] is not None:
        #     uat_names[uat_id] = uat_names[uat_id] + [alt_name.lower().strip() for alt_name in entry['altNames']]

	# sort by key
    uat_names = dict(sorted(uat_names.items()))

    return uat_names

def load_model_pipeline(pretrained_model_name_or_path=None, revision=None, tokenizer_model_name_or_path=None):
    """
    Load the model and tokenizer for the classification task, as well as the
    label mappings. Returns the model, tokenizer, and label mappings as a
    dictionary.

    Parameters
    ----------
    pretrained_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the model to load. If None, then reads from the 
        config file.
    revision : str (optional) (default=None) Specifies the revision of the model
    tokenizer_model_name_or_path : str (optional) (default=None) Specifies the
        model name or path to the tokenizer to load. If None, then defaults to
        the pretrained_model_name_or_path.
    """
    # Define labels and ID mappings
    # labels = ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']
    # id2label = {i:c for i,c in enumerate(labels) }
    # label2id = {v:k for k,v in id2label.items()}
    uat_names = load_uat()

    # Define model and tokenizer
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL']
    if revision is None:
        revision = config['CLASSIFICATION_PRETRAINED_MODEL_REVISION']
    if tokenizer_model_name_or_path is None:
        tokenizer_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

    pipe = pipeline(task='sentiment-analysis',
                    model=pretrained_model_name_or_path,
					tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path, 
															model_max_length=512, 
															do_lower_case=False,
															),
					revision=revision,
					num_workers=1,
					batch_size=32,
					return_all_scores=True,
					truncation=True,
					)


    return pipe



# split tokenized text into chunks for the model
def input_ids_splitter(input_ids, window_size=510, window_stride=255):
    '''
    Given a list of input_ids (tokenized text ready for a model),
    returns a list with chuncks of window_size, starting and ending with the special tokens (potentially with padding)
    the chuncks will have overlap by window_size-window_stride
    '''
        
    # int() rounds towards zero, so down for positive values
    num_splits = max(1, int(len(input_ids)/window_stride))
    
    split_input_ids = [input_ids[i*window_stride:i*window_stride+window_size] for i in range(num_splits)]
    
    
    return(split_input_ids)


def add_special_tokens_split_input_ids(split_input_ids, tokenizer):
    '''
    adds the start [CLS], end [SEP] and padding [PAD] special tokens to the list of split_input_ids
    '''
    
    # add start and end
    split_input_ids_with_tokens = [[tokenizer.cls_token_id]+s+[tokenizer.sep_token_id] for s in split_input_ids]
    
    # add padding to the last one
    split_input_ids_with_tokens[-1] = split_input_ids_with_tokens[-1]+[tokenizer.pad_token_id 
                                                                       for _ in range(len(split_input_ids_with_tokens[0])-len(split_input_ids_with_tokens[-1]))]
    
    return(split_input_ids_with_tokens)

    
def batch_assign_SciX_categories(list_of_texts, tokenizer, model,labels,id2label,label2id, score_combiner='max', score_thresholds=None, window_size=510,  window_stride=500):
    '''
    Given a list of texts, assigns SciX categories to each of them.
    Returns two items:
        a list of categories of the form [[cat_1,cat2], ...] (the predicted categories for each text in the input list, texts can be in multiple categories)
        a list of detailed scores of the form [(ast_score, hp_score ...) ...] (the predicted scores for each category for each text in the input list). The scores are in order ['Astronomy', 'Heliophysics', 'Planetary Science', 'Earth Science', 'NASA-funded Biophysics', 'Other Physics', 'Other', 'Text Garbage']

    
    Other than the required list of texts, this functions has a number of optional parameters to modify its behavior.
    pretrained_model_name_or_path: defaults to 'adsabs/astroBERT', but can replaced with a path to a different finetuned categorizer
    revision: defaults to 'SCIX-CATEGORIZER' so that huggigface knows which version of astroBERT to download. Probably never needs to be changed.
    score_combiner: Defaults to 'max'. Can be one of: 'max', 'mean', or a custom lambda function that combines a list of scores per category for each sub-sample into one score for the entire text (this is needed when the text is longer than 512 tokens, the max astroBERT can handle).
    score_thresholds: list of thresholds that scores in each category need to surpass for that category to be assigned. Defaults are from testing.
    
    # splitting params, to handle samples longer than 512 tokens.
    window_size = 510
    window_stride = 500    
    '''
    
    
    # optimal default thresholds based on experimental results
    if score_thresholds is None:
        score_thresholds = [0.0 for _ in range(len(labels)) ]

    
    # import pdb; pdb.set_trace()
    
    list_of_texts_tokenized_input_ids = tokenizer(list_of_texts, add_special_tokens=False)['input_ids']
    # list_of_texts_tokenized_input_ids = tokenizer(list_of_texts, add_special_tokens=False)['input_ids'][0]
    
    # split
    list_of_split_input_ids = [input_ids_splitter(t, window_size=window_size, window_stride=window_stride) for t in list_of_texts_tokenized_input_ids]
    
    # add special tokens
    list_of_split_input_ids_with_tokens = [add_special_tokens_split_input_ids(s, tokenizer) for s in list_of_split_input_ids]
    
    
    # list to return
    list_of_categories = []
    list_of_scores = []
    
    # forward call
    with no_grad():
        # for split_input_ids_with_tokens in tqdm(list_of_split_input_ids_with_tokens):
        for split_input_ids_with_tokens in list_of_split_input_ids_with_tokens:
            # make predictions
            predictions = model(input_ids=tensor(split_input_ids_with_tokens)).logits.sigmoid()
            
            # combine into one prediction
            if score_combiner=='mean':
                prediction = predictions.mean(dim=0)
            elif score_combiner=='max':
                prediction = predictions.max(dim=0)[0]
            else:
                # should be a custom lambda function
                prediction = score_combiner(predictions)
            
            list_of_scores.append(prediction.tolist())
            # filter by scores above score_threshold
            list_of_categories.append([id2label[index] for index,score in enumerate(prediction) if score>=score_thresholds[index]])
    
    return(list_of_categories, list_of_scores)
    

def score_record(record):
    """
    Provide classification scores for a record using the following
        categories:
            0 - Astronomy
            1 - HelioPhysics
            2 - Planetary Science
            3 - Earth Science
            4 - Biological and Physical Sciences
            5 - Other Physics
            6 - Other
            7 - Garbage

    Parameters
    ----------
    records_path : str (required) (default=None) Path to a .csv file of records

    Returns
    -------
    records : dictionary with the following keys: bibcode, text,
                categories, scores, and model information
    """
    # Load model and tokenizer
    model_dict = load_model_and_tokenizer()

    text = f"{record['title']} {record['abstract']}"

    # Classify record
    record['categories'], record['scores'] = classifier.batch_assign_SciX_categories(
                                [text],model_dict['tokenizer'],
                                model_dict['model'],model_dict['labels'],
                                model_dict['id2label'],model_dict['label2id'])

    # Because the classifier returns a list of lists so it can batch process
    # Take only the first element of each list
    record['categories'] = record['categories'][0]
    record['scores'] = record['scores'][0]

    # Append model information to record
    # record['model'] = model_dict['model']
    record['model'] = model_dict


    # print("Record: {}".format(record['bibcode']))
    # print("Text: {}".format(record['text']))
    # print("Categories: {}".format(record['categories']))
    # print("Scores: {}".format(record['scores']))

    return record

def classify_record_from_scores(record):
    """
    Classify a record after it has been scored. 

    Parameters
    ----------
    record : dictionary (required) (default=None) Dictionary with the following
        keys: bibcode, text, categories, scores, and model information

    Returns
    -------
    record : dictionary with the following keys: bibcode, text, categories,
        scores, model information, and Collections
    """

    # Fetch thresholds from config file
    thresholds = config['CLASSIFICATION_THRESHOLDS']
    # print('Thresholds: {}'.format(thresholds))


    scores = record['scores']
    categories = record['categories']
    # max_score_index = scores.index(max(scores))
    # max_category = categories[max_score_index]
    # max_score = scores[max_score_index]

    meet_threshold = [score > threshold for score, threshold in zip(scores, thresholds)]

    # Extra step to check for "Earth Science" articles miscategorized as "Other"
    # This is expected to be less neccessary with improved training data
    if config['ADDITIONAL_EARTH_SCIENCE_PROCESSING'] is True:
        # print('Additional Earth Science Processing')
        # import pdb;pdb.set_trace()
        if meet_threshold[categories.index('Other')] is True:
            # If Earth Science score above additional threshold
            if scores[categories.index('Earth Science')] \
                    > config['ADDITIONAL_EARTH_SCIENCE_PROCESSING_THRESHOLD']:
                meet_threshold[categories.index('Other')] = False
                meet_threshold[categories.index('Earth Science')] = True

    # Append collections to record
    record['collections'] = [category for category, threshold in zip(categories, meet_threshold) if threshold is True]
    record['earth_science_adjustment'] = config['ADDITIONAL_EARTH_SCIENCE_PROCESSING']

    return record


