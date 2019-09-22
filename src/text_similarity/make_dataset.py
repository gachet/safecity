import logging
import os

import click
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector


def drop_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform column-level and entry-level drops
    """

    # drop duplicates that have same comment and same place
    df = df.drop_duplicates(subset=['LATITUDE', 'LONGITUDE', 'DESCRIPTION'])

    # drop entries where comments is left blank
    df.dropna(subset=['DESCRIPTION'], inplace=True)

    # drop 'More Info' column (too many nulls)
    df.drop('More Info', axis=1, inplace=True)

    return df.reset_index(drop=True)


def prune_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple features such as number of words, length of characters in order to
    prune dataset more
    """
    # create number of words column
    df['num_words'] = df['DESCRIPTION'].apply(lambda x: len(x.split()))

    # from exploratpry data analysis, sentences form with 4 or more words,
    # drop entries with less than 3 words
    df = df[df['num_words'] > 3]

    # load in spacy & language detector
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    # simple function to return detected language
    def get_lang(desc):
        doc = nlp(desc)
        return doc._.language['language']

    df['lang'] = df['DESCRIPTION'].apply(get_lang)

    # hindi and indonesian languages are most common, remove these for better nlp processing
    df = df[~df['lang'].isin(['hi', 'id'])]

    return df.reset_index(drop=True)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs top-level data processing functions to turn raw data from into interim data
    """
    logger = logging.getLogger(__name__)
    logger.info('Performing top-level data processing from raw data to be saved to interim')

    assert os.path.isfile(input_filepath), "Path to data doesn't exist!"

    logger.info('Reading in data...')
    df = pd.read_csv(input_filepath, dtype={'#': object})

    logger.info('Dropping duplicates and nulls...')
    df = drop_entries(df)

    logger.info('Pruning data by discarding entries with minimal descriptions'
                ' and non-english languages...')
    df = prune_data(df)

    logger.info('Saving interim data to interim directory')
    df.to_pickle(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
