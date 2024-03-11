import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_args, load_config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_csvs(path_to_training, path_to_testing, concat, test_size):

    if concat:
        train_df, test_df = concat_df(
            path_to_training, path_to_testing, test_size)
    else:
        train_df = pd.read_csv(path_to_training)
        test_df = pd.read_csv(path_to_testing)

    return train_df, test_df


def concat_df(df1_path: str, df2_path: str, test_size: float) -> tuple:
    """
    Concatenates two dataframes and splits the combined dataframe into
    train and test sets.

    Parameters:
    df1_path (str): The file path of the first dataframe.
    df2_path (str): The file path of the second dataframe.
    test_size (float): The proportion of the dataframe to
                        include in the test set.

    Returns:
    tuple: A tuple containing four dataframes: X_train, X_test,
            y_train, y_test.
    """

    df = pd.concat([pd.read_csv(df1_path), pd.read_csv(df2_path)])
    return train_test_split(
        df, test_size=test_size, shuffle=True, random_state=42)


def preprocess_df(df, drop_social_pressure, drop_dups):

    if drop_social_pressure:
        df = df.drop(columns=['6-SOCIAL', '7-PRESSURE'])

    if drop_dups:
        df = df.drop_duplicates()

    return df


def output_df(training_df, testing_df, config):
    training_path = config.get('output_training_df_path') + \
          'drop_social_pressure_' + str(config.get('drop_social_pressure')) + \
            '_drop_dups_' + str(config.get('drop_dups')) + '.csv'

    testing_path = config.get('output_testing_df_path') + \
        'drop_social_pressure_' + str(config.get('drop_social_pressure')) + \
        '_drop_dups_' + str(config.get('drop_dups')) + '.csv'

    training_df.to_csv(training_path)
    testing_df.to_csv(testing_path)
    logger.info(f"Dataframes output to {training_path} and {testing_path}")


def build_vocab(training_df):

    vocab = {'<start>' : 1,
             '<end>' : 2,
             '<unk>' : 3}

    for _, row in training_df.iterrows():
        for token in row.text.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_text(df, vocab):
    df['encoded'] = ''
    for idx, row in df.iterrows():
        text = row.text.split()
        text.insert(0, '<start>')
        text.append('<end>')
        text = [str(vocab[token]) if token in vocab else str(vocab['<unk>']) for token in text]
        text = ' '.join(text)
        df.at[idx, 'encoded'] = text
    return df


def preprocess_lstm(train_df, test_df):
    vocab = build_vocab(train_df)
    train_df = encode_text(train_df, vocab)
    test_df = encode_text(test_df, vocab)
    return train_df, test_df, vocab


def import_data(config: dict):
    vocab = None
    training_df, testing_df = load_csvs(
        config.get('path_to_training'),
        config.get('path_to_testing'),
        config.get('concat'),
        config.get('test_size')
        )

    training_df = preprocess_df(
        training_df, config.get('drop_social_pressure'),
            config.get('drop_dups'))
    testing_df = preprocess_df(
        testing_df, config.get('drop_social_pressure'),
            config.get('drop_dups'))
    
    if not config.get('use_pretrained'):
        training_df, testing_df, vocab = preprocess_lstm(training_df, testing_df)
        
    # labels = training_df.columns[1:].tolist()

    if config.get('verbose'):
        print(f'Num Training {len(training_df)}')
        print(f'Num Evaluation {len(testing_df)}')

    if config.get('output_dfs'):
        output_df(training_df, testing_df, config)

    return training_df, testing_df, vocab


if __name__ == '__main__':
    args = get_args()
    config = load_config(args.config_path)
    train_df, test_df, label_columns = import_data(config)
