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
        uat_dict = json.load(f)

    return uat_dict

def load_uat_names():
    # build the dict that matches UAT ID (numbers) to common names
    uat_dict = load_uat()

    uat_names = {}
    for entry in uat_dict.keys():
        uat_names[entry] = uat_dict[entry]['name'].lower().strip()
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
    # Define UAT labels and names
    uat_names = load_uat_names()


    # Define model and tokenizer
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL']
    if revision is None:
        revision = config['CLASSIFICATION_PRETRAINED_MODEL_REVISION']
    if tokenizer_model_name_or_path is None:
        tokenizer_model_name_or_path = config['CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER']

    print(f'Loading model: {pretrained_model_name_or_path}')

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

