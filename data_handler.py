from typing import Any, Union

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import paths


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

    R: coo_matrix = coo_matrix((predictions, (user_ids, item_ids)))

    return R


def preprocess_data(csv_file, csv_to):
    """ Preprocess the original data given to us

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


def split_original_data():
    total_dataset = pd.read_csv(paths.total_dataset_location)
    num_col = (0.75 * total_dataset.groupby('col_id')['row_id'].count()).astype(int)

    train_dataset = pd.DataFrame(columns=['Id', 'Prediction', 'row_id', 'col_id'])
    test_dataset = pd.DataFrame(columns=['Id', 'Prediction', 'row_id', 'col_id'])
    for i in range(1, 1001):
        current_rows = total_dataset.loc[total_dataset['col_id'] == i]
        train_dataset = train_dataset.append(current_rows.iloc[0:num_col.iloc[i-1]])
        train_dataset = train_dataset.append(current_rows.iloc[-1])
        test_dataset = test_dataset.append(current_rows.iloc[num_col.iloc[i-1]: -1])
        if i % 100 == 0:
            print(i)
    print(train_dataset.groupby('col_id')['row_id'].count())

    nrows = total_dataset.shape[0]
    train_dataset = total_dataset.iloc[0:int(nrows*0.7)]
    train_dataset = train_dataset.append(total_dataset.iloc[-1])
    print(train_dataset.iloc[-1])
    train_dataset.to_csv(paths.train_data_location, index=False)

    test_dataset = total_dataset.loc[int(nrows*0.7):(nrows-1)]
    test_dataset.to_csv(paths.test_data_location, index=False)


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
