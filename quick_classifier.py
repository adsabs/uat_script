#!/usr/bin/env python
"""
"""

# __author__ = 'tsa'
# __maintainer__ = 'tsa'
# __copyright__ = 'Copyright 2024'
# __email__ = 'ads@cfa.harvard.edu'
# __status__ = 'Production'
# __credit__ = ['T. Allen']
# __license__ = 'MIT'

import os
import csv
import argparse
import sys
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from uat_script import classifier
from harvest_solr import harvest_solr
from uat_script.classifier import load_model_pipeline, load_uat, load_uat_names

import heapq

# # ============================= INITIALIZATION ==================================== #

from adsputils import setup_logging, load_config
proj_home = os.path.realpath(os.path.dirname(__file__))
global config
config = load_config(proj_home=proj_home)
logger = setup_logging('run.py', proj_home=proj_home,
                        level=config.get('LOGGING_LEVEL', 'INFO'),
                        attach_stdout=config.get('LOG_STDOUT', False))

uat_pipeline = load_model_pipeline()
uat_names = load_uat_names()
full_uat_dict = load_uat()

# =============================== FUNCTIONS ======================================= #

def write_batch_to_tsv(batch, header, filename, mode='a', include_header=False):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if include_header:
            writer.writerow(header)
        writer.writerows(batch)

def top_k_scores(scores, k):
    return(heapq.nlargest(k, scores, key=lambda x: x['score']) )
        
def bottom_k_scores(scores, k):
    return(heapq.nsmallest(k, scores, key=lambda x: x['score']) )
    
def paths_scores(uat_id, pred_scores, uat_dict):
    '''
    function that returns the scores along each path to uat_id, given the pred_score (output of the model pipeline)
    ex: pred_scores = [{'label:'0', 'score':0.987}, {'label:'2'...}, ...]
    (note pipeline returns one list per input), so this is an element of the output 
    returns: [{'path':path_list, 'score':score}, ...] 
    ex: [{'path': ['1503', '709', '1476'], 'scores': [0.0009542001644149, 2.0110092009417717e-06, 1.2095058536942815e-06]}, {'path': ['1503', '1476'], 'scores': [0.0009542001644149, 1.2095058536942815e-06]}]
    note that the path is sorted from leaf -> .. -> root, leave being the predicted keyword
    '''
    # get paths
    # import pdb; pdb.set_trace()
    all_paths = uat_dict[uat_id]['paths']
    all_paths_scores = []
    
    # rework the pred score to be more accessible
    scores_dict = {str(d['label']):d['score'] for d in pred_scores}
    
    for path in all_paths:
        # print([scores_dict[uat_id] for uat_id in path])
        # import pdb; pdb.set_trace()
        scores_in_path = [scores_dict[uat_id] for uat_id in path]
        

        all_paths_scores.append({'path': path, 'scores':scores_in_path})
    
    return(all_paths_scores)

def argmax(a):
    return max(range(len(a)), key=lambda x : a[x])
        
# =============================== MAIN ======================================= #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process user input.')


    parser.add_argument('-r',
                        '--records',
                        dest='records',
                        action='store',
                        help='Path to list of bibcodes to process')
    parser.add_argument('-p',
                        '--preserve_filename',
                        dest='preserve_filename',
                        action='store_true',
                        help='Set to apply input filename to output')
    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        help='Print individual record information to screen as records are being processed')

    args = parser.parse_args()

    if args.preserve_filename:
        preserve_filename = True
    else:
        preserve_filename = False

    if args.records:
        records_path = args.records
        if preserve_filename:
            out_path = records_path.replace('.csv', '_uat_keywords.tsv')
        else:
            out_path = config['OUTPUT_FILE']
        print(f'Reading in {records_path} may take a minute for large input files.')
        print(f'Will write output to {out_path}.')
    else:
        print("Please provide a path to a .csv file with records to process.")
        exit()

    if args.verbose:
        verbose = True
    else:
        verbose = False

    output_idx = 0
    output_list = []
    output_batch = 500
    header = 'bibcode,uat_branch,uat_id'

    with open(records_path, 'r') as f:
        bibcodes = f.read().splitlines()

    # If first line is 'bibcode' remove it
    if bibcodes[0]=='bibcode':
        bibcodes = bibcodes[1:]

    print('Classifying records...')
    while output_idx < len(bibcodes):

        # Harvest Title and Abstract from Solr
        bibcode_batch = bibcodes[output_idx:output_idx+output_batch]
        records = harvest_solr(bibcode_batch, start_index=0, fields='bibcode, title, abstract')
        if len(records) == 0:
            sys.exit('No records returned from harvesting Solr - exiting')

        for index, record in enumerate(records):

            title = ''
            abstract = ''
            if 'title' in record:
                title = record['title']
                if isinstance(title, list):
                    title = title[0]
                    record['title'] = title

            if 'abstract' in record:
                abstract = record['abstract']

            text = str(title) + ' ' + str(abstract)

            if verbose:
                print()
                print(f'Bibcode: {record["bibcode"]}')
                print(f'Title: {title}')
                print(f'Abstract: {abstract}')

            # Inference
            scores = uat_pipeline(text)
            scores = scores[0]

            # Thresholding
            threshold = config['CLASSIFICATION_THRESHOLD']

            try:
                keywords = [s for s in scores if s['score'] >= threshold]
                for word in keywords:
                    word['category'] = uat_names.get(word['label'],'Unknown')            
                keywords = sorted(keywords, key=lambda x: x['score'], reverse=True)
            except:
                keywords = None

            if keywords is not None:
                keyword_list = [kw['category'] for kw in keywords]
                score_list = [kw['score'] for kw in keywords]

                for keyword in keywords:
                    if verbose:
                        print()
                        print('Keyword')
                        print(keyword)
                    # Get scores for each path to the keyword
                    paths_scores_list = paths_scores(str(keyword['label']), scores, full_uat_dict)
                    # Get top score
                    mean_scores = [float(sum(s['scores']))/float(len(s['scores'])) for s in paths_scores_list]
                    max_mean_score_ind = argmax(mean_scores)  
                    # reverse the path to go from root -> leaf
                    path = paths_scores_list[max_mean_score_ind]['path'][::-1]
                    path = [uat_names.get(p, 'Unknown') for p in path]
                    # path = '/'.join(path)
                    path = path[-1] # get the last element of the path
                    # Not record path at moment
                    record_output = [record['bibcode'], path, keyword['label']]
                    output_list.append(record_output)

        if output_idx == 0:
            include_header = False
            mode = 'a'
        else:
            include_header = False
            mode = 'a'
        write_batch_to_tsv(output_list, header.split(','), out_path, mode=mode, include_header=include_header)
        output_list = []
        print(f"Processed {output_idx+index+1} records")

        output_idx += output_batch 


    print("Done")
    print(f"Results saved to {out_path}")
