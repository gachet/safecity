import json
import logging
import os
from typing import List, Dict

import click
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs simple lemmatization from NLTK wordnet
    """
    nltk.download('wordnet')

    lemmer = nltk.stem.WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]

    df['lemmed'] = df[['DESCRIPTION']].apply(LemTokens)

    return df


def compute_similarities(df: pd.DataFrame) -> np.ndarray:
    """
    Performs custom stop words additions, tokenization, TF-IDF vectorization
    and cosine similarity computation
    """
    # first ensure DataFrame column is present
    assert 'lemmed' in df.columns, "Lemmatized column not present in DataFrame!"

    # adding common stop words that don't provide extra meaning
    stop_words_added = ['boy', 'girl', 'man']
    combined_stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_added)

    # instantiate Vectorizer class and compute similarities
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0,
                         stop_words=combined_stop_words)
    tfidf_matrix = tf.fit_transform(df['lemmed'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_similarities


def get_similar_items(cosine_similarities: np.ndarray, num_items: int = 5,
                      drop_first: bool = True) -> List:
    """
    Get's top similar indices for each profile description
    """
    # sort the cosine_similarities, returns indices of sorted array
    cosine_sorted = cosine_similarities.argsort()

    # almost always drop the first in sorted array, likely the same comment
    # next section should return a list of tuples going through each row of the sorted array
    # the first entry in the tuple is the similar indices ([::-1] arranges in descending order)
    # the second entry in the tuple is the corresponding similarity values [0,1] for each similar index
    if drop_first:
        similar_items = [(j[::-1][1:1 + num_items],
                          cosine_similarities[i][j[::-1]][1:1 + num_items]) for i, j in
                         enumerate(cosine_sorted)]
    else:
        similar_items = [(j[::-1][:num_items],
                          cosine_similarities[i][j[::-1]][:num_items]) for i, j in
                         enumerate(cosine_sorted)]

    return similar_items


def generate_mappings(similar_items: List, df: pd.DataFrame) -> pd.DataFrame:
    """
    Get similar descriptions into DataFrame columns
    """
    # get the similar indices
    similar_indices = [i[0] for i in similar_items]

    # for each row, get the values of each similar index and its similarity score
    df['similar_desc'] = [(i, df['DESCRIPTION'].iloc[i].values.tolist()) for i in similar_indices]
    df['similarity_score'] = [i[1].tolist() for i in similar_items]

    return df


def output_dict(df: pd.DataFrame) -> Dict:
    """
    Transforms the DataFrame into json friendly format that can be inserted into NoSQL db
    """
    # rename columns
    df.rename(columns={'#': 'reportNumber',
                       'INCIDENT DATE': 'incidentDate',
                       'LOCATION': 'location',
                       'DESCRIPTION': 'description',
                       'similar_desc': 'similarReports',
                       'similarity_score': 'similarityScore'
                       }, inplace=True)

    # take only relevant columns
    df = df[['reportNumber', 'incidentDate', 'location', 'description',
             'similarReports', 'similarityScore']]

    # create a dictionary from 'similarReports' list
    df['similarDict'] = df['similarReports'].apply(lambda x: [
        df[['reportNumber', 'incidentDate', 'location', 'description']].iloc[i].to_dict() for i in x[0]])

    # create a dictionary of all columns
    df_dict = df[['reportNumber', 'incidentDate', 'location', 'description',
                  'similarityScore', 'similarDict']].to_dict('r')

    # json-friendly formatting
    for report_dict in df_dict:
        report_dict['similarReports'] = report_dict.pop('similarDict')
        [k.update({'similarityScore': v}) for k, v in
         zip(report_dict['similarReports'], report_dict['similarityScore'])]
        report_dict.pop('similarityScore', None)

    return df_dict


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Takes interim data and performs NLP tasks and saves dictionary of similar comments in json
    """
    logger = logging.getLogger(__name__)
    logger.info('Performing top-level data processing from raw data to be saved to interim')

    assert os.path.isfile(input_filepath), "Path to data doesn't exist!"

    logger.info('Reading in data...')
    df = pd.read_pickle(input_filepath)

    logger.info('Performing lemmatization...')
    df = lemmatize(df)

    logger.info('Computing similarities...')
    cos_similarities = compute_similarities(df)

    logger.info('Sorting similarities...')
    similar_items = get_similar_items(cos_similarities)

    logger.info('Generating similar descriptions...')
    df = generate_mappings(similar_items, df)

    logger.info('Transforming to dictionary and writing as json...')
    similar_dict = output_dict(df)

    # save the dictionary in json under models directory
    with open(output_filepath, 'w') as f:
        json.dump(similar_dict, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
