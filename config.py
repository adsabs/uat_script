
API_URL = "https://devapi.adsabs.harvard.edu/v1" # ADS API URL
API_TOKEN = ""

UAT_JSON_PATH = 'full_UAT_dict.json'

CLASSIFICATION_PRETRAINED_MODEL = "adsabs/nasa-smd-ibm-v0.1_UAT_Labeler"
CLASSIFICATION_PRETRAINED_MODEL_REVISION = None
CLASSIFICATION_PRETRAINED_MODEL_TOKENIZER = None

# Chosen because in testing gives roughly 3 predictions per sample
CLASSIFICATION_THRESHOLD = 0.136

OUTPUT_FILE = 'uat_script/tests/stub_data/stub_bibcodes_uat_keywords_classified.tsv'



