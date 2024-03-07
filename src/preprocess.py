import pandas as pd
from sklearn.model_selection import train_test_split

def load_csvs(path_to_training, path_to_testing, concat, test_size):
    
    if concat:
        train_df, test_df = concat_df(path_to_training, path_to_testing, test_size)
    else:
        train_df = pd.read_csv(path_to_training)
        test_df = pd.read_csv(path_to_testing)
    
    return train_df, test_df

def concat_df(df1_path: str, df2_path: str, test_size: float) -> tuple:
    """
    Concatenates two dataframes and splits the combined dataframe into train and test sets.

    Parameters:
    df1_path (str): The file path of the first dataframe.
    df2_path (str): The file path of the second dataframe.
    test_size (float): The proportion of the dataframe to include in the test set.

    Returns:
    tuple: A tuple containing four dataframes: X_train, X_test, y_train, y_test.
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

def import_data(config:dict):

    training_df, testing_df = load_csvs(
        config.get('path_to_training'), 
        config.get('path_to_testing'), 
        config.get('concat'), 
        config.get('test_size')
        )

    training_df = preprocess_df(training_df, config.get('drop_social_pressure'), config.get('drop_dups'))
    testing_df = preprocess_df(testing_df, config.get('drop_social_pressure'), config.get('drop_dups'))
    
    labels = training_df.columns[1:].tolist()

    if config.get('verbose'):
        print(f'Num Training {len(training_df)}')
        print(f'Num Evaluation {len(testing_df)}')

    return training_df, testing_df, labels

if __name__ == '__main__':
    config = {
        'path_to_training': 'training.csv',
        'path_to_testing': 'testing.csv',
        'DROP_SOCIAL_PRESSURE': True,
        'concat': True,
        'drop_dups': True,
        'test_size': 0.2,
        'verbose': True
    }

    train_df, test_df, label_columns = import_data(config)