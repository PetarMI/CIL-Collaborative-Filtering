import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


def read_data(data_file: str) -> pd.DataFrame:
    """ Load the data from a file to a pandas dataframe

    :param data_file: String pointing to the csv file
    :return: pandas dataframe containing the data
             Id | Prediction | row_id | col_id
             ---------------------------------
             id |     4      |   44   |    1
             id |     3      |   61   |    1
             id |     4      |   67   |    1
             id |     3      |   72   |    1
    """
    df_data = pd.read_csv(data_file)
    return df_data


def data_as_matrix(df_data: pd.DataFrame) -> coo_matrix:
    """ Take the data in the form of a pandas dataframe and convert it
        to a sparse matrix
    """
    user_ids = df_data['row_id'] - 1
    item_ids = df_data['col_id'] - 1
    predictions = df_data['Prediction']

    R: coo_matrix = coo_matrix((predictions, (user_ids, item_ids)), shape=(10000, 1000))

    return R


def df_as_array(df_data: pd.DataFrame) -> np.ndarray:
    """ Convert the given dataset to an array of tuples
        where each tuple contains a single entry for user-movie rating

    :param df_data: Data to convert
    :return train_data: Data as an array
    """
    user_ids = df_data["row_id"] - 1
    movie_ids = df_data["col_id"] - 1
    ratings = df_data["Prediction"]

    train_data = np.asarray(list(zip(user_ids, movie_ids, ratings)))
    return train_data


# TODO annotate return type
def split_original_data(df_data: pd.DataFrame, test_fraction: float):
    """ Split the original data into train and test data
        with ratio 80:20

    :param df_data: The data we are splitting
    :param test_fraction: Fraction of the total dataset we use for testing
    :return: a dictionary for both train and test data
    """
    test_data: pd.DataFrame = df_data.sample(frac=test_fraction)
    test_data.sort_index(inplace=True)
    train_data: pd.DataFrame = df_data.drop(test_data.index)

    return { "train_data" : train_data,
             "test_data" : test_data }


def preprocess_data(csv_file, csv_to):
    """ Preprocess the original data given to us (one-time function)

    :param csv_file: The file containing the original data
    :param csv_to: File which we save the processed data to
    :return:
    """
    df_read = pd.read_csv(csv_file)
    id_string = df_read['Id'].str.split('_')
    df_read['row_id'] = id_string.str[0].str[1:].astype('int32')
    df_read['col_id'] = id_string.str[1].str[1:].astype('int32')

    df_read.to_csv(csv_to, index=False)
    print(csv_file.split('/')[-1] + ' has been processed!')
    return df_read


# Write the results from matrix to .csv and matches with the sample submission
def write_submission(A, **kwarg):
    src = kwarg.get('src',
                    './data/sampleSubmission.csv')
    dst = kwarg.get('dst',
                    './data/submission.csv')

    df_read = pd.read_csv(src)
    id_string = df_read['Id'].str.split('_')
    row_ids = id_string.str[0].str[1:].astype('int32')
    col_ids = id_string.str[1].str[1:].astype('int32')
    df_read['Prediction'] = A[row_ids - 1, col_ids - 1].T
    df_read.to_csv(dst, index=False)
    print('Total number of entries in the submission file: %d'
          % np.shape(df_read)[0])

    return None


def log(logger, msg: str, to_file: bool):
    if(to_file):
        logger.info(msg)
    else:
        print(msg)
