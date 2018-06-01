total_dataset_location = './data/data_train_post.csv'
train_data_location = "./data/bootstrap_train.csv"
test_data_location = "./data/bootstrap_test.csv"

num_rows = 10000
num_cols = 1000

num_users = 10000
num_movies = 1000

test_data_fraction = 0.2

min_rating = 1.0
max_rating = 5.0

# Learning rate
learning_rate = 0.001
# regularizer
lambda_term = 0.02

sgd_max_iteration = 1000

# indices for samples in the form of array of tuples
user_id = 0
movie_id = 1
rating_id = 2
