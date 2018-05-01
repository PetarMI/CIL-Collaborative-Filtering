import pandas as pd

# parse the row/colomn indices from the given csv file 
# and write the result into a new csv file
def csv_parse(csv_file, csv_to):    
    df_read = pd.read_csv(csv_file)
    id_string = df_read['Id'].str.split('_')
    df_read['row_id'] = id_string.str[0].str[1:].astype('int32')
    df_read['col_id'] = id_string.str[1].str[1:].astype('int32')
    
    df_read.to_csv(csv_to, index=False)
    print(csv_file.split('/')[-1]+' has been processed!')
    return df_read

# Write the results from matrix to .csv and matches with the samplesubmission
def write_submission(A, **kwarg):
    
    src = kwarg.get('src',
                    './cil-collab-filtering-2018/sampleSubmission.csv')
    dst = kwarg.get('dst',
                    './cil-collab-filtering-2018/submission.csv')
    
    df_read = pd.read_csv(src)
    id_string = df_read['Id'].str.split('_')
    row_ids = id_string.str[0].str[1:].astype('int32')
    col_ids = id_string.str[1].str[1:].astype('int32')
    df_read['Prediction'] = A[row_ids-1, col_ids-1].T
    df_read.to_csv(dst, index=False)
    print('Total number of entries in the submission file: %d' 
          % np.shape(df_read)[0])
    
    return None