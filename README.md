# CIL-Collaborative-Filtering
Repo for the CIL group project on Recommender Systems and Collaborative FIltering using SVD

### Updates after 2nd meeting on 30th-Apr

1. Wrap up the .csv file processing functions into helper.py, which contains two methods:
  
  - csv_parse(): read in the original csv file and parse the row/col index, then save a new csv with row/col ids for later use.
  - write_submission(): given a matrix with predicted results, the function write out a csv file named "submission.csv" for submission
  
  - **TODO:** creat another method which can automaticlly submit the submission file to Kaggle. This can be done with the help of Kaggle API.
  
2. Optimised the creation of baseline solution

  - Calculating the average of the non-zero elements in each colomn of matrix A is handled by a vectorized way, which is more efficient.
  
  - **TODO:** wrap up the baseline solution creation into a method
  
3. Try to finish SGD algorithm for making prediction

  - The SGD algorithm in Step-3 is not finished yet.  
