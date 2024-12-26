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

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from uat_script import classifier
from harvest_solr import harvest_solr
from uat_script.classifier import load_model_pipeline, load_uat

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
uat_names = load_uat()

# =============================== FUNCTIONS ======================================= #

def write_batch_to_tsv(batch, header, filename, mode='w', include_header=True):
    with open(filename, mode, newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        if include_header:
            writer.writerow(header)
        writer.writerows(batch)

def top_k_scores(scores, k):
    return(heapq.nlargest(k, scores, key=lambda x: x['score']) )
        
def bottom_k_scores(scores, k):
    return(heapq.nsmallest(k, scores, key=lambda x: x['score']) )

def get_keyword_hierarchy(keyword):
    hierarchy_paths = []

    with open(config['UAT_CSV_PATH'], 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)

        header = next(reader)

        for row in reader:
            row = [word.lower() for word in row]
            if keyword in row:
                idx = row.index(keyword)

                path = row[:idx+1]
                hierarchy_paths.append(path)

    # Take only unique paths, allow different paths to same concept
    unique_tuples = set(tuple(sublist) for sublist in hierarchy_paths)
    output_list = [list(t) for t in unique_tuples]
    output_list = ['/'.join(sublist) for sublist in output_list]

    return output_list
    

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
                    hierarchy = get_keyword_hierarchy(keyword['category'])
                    for path in hierarchy:
                        record_output = [record['bibcode'], path, keyword['label']]
                        output_list.append(record_output)

        if output_idx == 0:
            include_header = True
            mode = 'w'
        else:
            include_header = False
            mode = 'a'
        write_batch_to_tsv(output_list, header.split(','), out_path, mode=mode, include_header=include_header)
        output_list = []
        print(f"Processed {output_idx+index+1} records")

        output_idx += output_batch 


    print("Done")
    print(f"Results saved to {out_path}")
