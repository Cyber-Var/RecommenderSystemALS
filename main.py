import numpy as np
import csv

# TODO: remove
import os
import time

"""
- THE TASK: "code a large-scale matrix factorisation recommender system algorithm to
  train and then predict ratings for a large (20M) set of items"

- UNDERSTANDING THE TASK:
  1. Matrix factorization = a section of linear algebra that focuses on mathematical operations that work by factorizing 
     a matrix into a product of matrices
  2. Matrix factorization recommender system = recommender system that factorizes the user-item ratings matrix into 
     a product of a user matrix and an item matrix, where:
         2.1. User matrix = matrix with latent factors (preferences) of each user.
         2.2. Item matrix = matrix with latent factors of each item.
         2.3. Latent factors = characteristics (features) of users/items. By learning these factors of the training set,
              they can be further used to  predict ratings on the testing set.

- CHOICE OF THE ALGORITHM:
  As found in literature, the most popular matrix factorization recommender algorithms are: ALS and SVD [1, 2, 3]. 
  However, ALS tends to outperform SVD on large datasets in terms of error metrics [1, 4, 5].
  Furthermore, SVD often struggles with large and sparse datasets (while our task is to train on 20M instances of 
  training data) [4, 6], whereas the memory efficiency of ALS is higher and it is better at handling missing values.
  Therefore, the ALS algorithm is implemented in this code [5, 6].
  
- UNDERSTANDING THE ALS ALGORITHM:
  The Alternating Least Squares (ALS) algorithm works by the following steps:
  1. Two matrices are created and initialized with random values - user matrix (U) and item matrix (I), where the user 
     matrix stores the latent factors of each user, and the item matrix stores the latent factors of each item, as 
     described before. 
     Each row of the user matrix represents a user, and each row in the item matrix represents an item. 
     The number of columns in these matrices equals to the number of latent factors, where the optimal number will be 
     identified during hyperparameter tuning.
     The shape of the user and item matrices is therefore as follows:
     Shape of U = (number of users, number of latent factors).
     Shape of I = (number of items, number of latent factors).
  2. Next comes the iterative step, where the algorithm employs an alternating process in which:
     Item matrix fix: the algorithm optimizes the user matrix, while the item matrix is kept constant.
     User matrix fix: the algorithm optimizes the item matrix, while the user matrix is kept constant.
     These alternating fixes are repeated for a number of iterations, where the optimal number of iterations will be 
     identified during hyperparameter tuning.
     The aim of each iteration is to reduce the error between the training data and the user/item matrices, by 
     minimizing the loss function, which formula is [6]:
         Loss = \sum_{(u,i) \in R} (R_{ui} - U_u \cdot I_i)^2, where:
                        R_{ui} = rating by user u for item i (from the training set).
                        U_u = latent factors of user u.
                        I_u = latent factors of item i.
                        R = set of ratings given in the training data.
  3. To prevent the algorithm from overfitting, regularization is implemented as the next step. An L2 regularization 
     term is included as part of the loss function, which then becomes:
         Loss = \sum_{(u,i) \in R} (R_{ui} - U_u \cdot I_i)^2 + \lambda I, where:
                        R_{ui} = rating by user u for item i (from the training set).
                        U_u = latent factors of user u.
                        I_u = latent factors of item i.
                        \lambda = the regularization term, where the optimal regularization term will be 
                                  identified during hyperparameter tuning.
                        R = set of ratings given in the training data.
                        I = an identity matrix with size num_latent_factors * num_latent_factors.
  
- IMPLEMENTATION:
  Pseudo_code:
      user_matrix = initialize_with_random_values()
      item_matrix = initialize_with_random_values()
      
      Repeat num_iterations times:
          # Item matrix I fixed, optimizing user matrix U
          For each user u:
              Get the set I_u # set of item factors of all items that user u rated
              Construct R_u   # set of ratings that user u made
              Update the user matrix U for user u while minimizing the loss:
                  U_u = (I_u^T I_u + \lambda I)^{-1} I_u^T R_u
                    
          # User matrix U fixed, optimizing item matrix I:
          For each item i:
              Construct U_i (user factors of users who rated i)
              Construct R_i (ratings received by i)
              Solve for i_i minimizing the loss:
                  I_i = (U_i^T U_i + \lambda I)^{-1} U_i^T R_i
      
  The actual implementation code (also commented with explanations) is provided after the REFERENCES section.

- HYPERPARAMETER TUNING:
  An automatic script was created for tuning the 3 hyperparameters:
      + number of latent factors,
      + number of iterations (how many times to repeat the alternating ALS process),
      + regularization term.
      
  The tuning process uses grid search, where the tested values are:
      + number of latent factors: [TODO, ...]
      + number of iterations: [TODO, ...]
      + regularization term: [TODO, ...]
  For the purpose of hyperparameter tuning, training set was randomly split into 90% for training and 10% for testing.
  
  Below are the hyperparameters that resulted in the best MAE score and are, therefore, used in the final version of
  the code:
      + number of latent factors: TODO
      + number of iterations: TODO
      + regularization term: TODO

- REFERENCES:
[1] A. Priyati, A. D. Laksito, and H. Sismoro, "The Comparison Study of Matrix Factorization on Collaborative Filtering 
    Recommender System," in 2022 5th International Conference on Information and Communications Technology (ICOIACT), 
    Yogyakarta, Indonesia, 2022, pp. 177-182, doi: 10.1109/ICOIACT55506.2022.9972018.
[2] Yu, H. F., Hsieh, C. J., Si, S., et al., "Parallel matrix factorization for recommender systems," Knowledge and 
    Information Systems, vol. 41, no. 3, pp. 793–819, 2014. [Online]. 
    Available: https://doi.org/10.1007/s10115-013-0682-2
[3] W. Nguyen, "A Literature Review of Collaborative Filtering Recommendation System using Matrix Factorization 
    algorithms," presented at the Conference, July 2021.
[4] Y. Koren, R. Bell, and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," 2009.
[5] Y. Zhou, D. Wilkinson, R. Schreiber, and R. Pan, "Large-scale Parallel Collaborative Filtering for the Netflix 
    Prize," 2008.
[6] E. Rosenthal, Y.-J. Lee, and V. Kuleshov, "Matrix Factorization Techniques for Recommender Systems," 2016.

"""


def read_dataset_from_file(filename):
    """
    Function for reading a dataset from a given file and storing it as a numpy array
    :param filename: the name of the file to read
    :return: numpy array with all contents of the file stored inside
    """
    dataset = []
    # Open the file for reading:
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        # Read the file line by line:
        for line in reader:
            # Split each line on commas and store it in array:
            row = [int(item) if item.isdigit() else float(item) if '.' in item else item for item in line[0].split(",")]
            dataset.append(row)
    # Return the numpy array:
    return np.array(dataset, dtype=object)


def get_unique_users_items(dataset1, dataset2):
    """
        Function that finds all unique users and all unique items within both the train set and test set,
        and returns them
        :param dataset1: train set
        :param dataset2: test set
        :return: 2 lists - one that stores all unique user ids, and one that stores all unique item ids
    """
    # Find unique user ids and item ids in datasets:
    unique_users1, unique_items1 = np.unique(dataset1[:, 0]), np.unique(dataset1[:, 1])
    unique_users2, unique_items2 = np.unique(dataset2[:, 0]), np.unique(dataset2[:, 1])

    # Union the user ids and item ids of both datasets:
    all_unique_users = list(set(unique_users1).union(set(unique_users2)))
    all_unique_items = list(set(unique_items1).union(set(unique_items2)))
    return all_unique_users, all_unique_items


def create_ratings_matrix(dataset, unique_users_set, unique_items_set):
    """
    Function that converts data read from file into a matrix, where rows are user ids, columns are item ids, and values
    are the ratings
    :param dataset: data read from one of the given files
    :param unique_users_set: list of all unique user ids found from given dataset
    :param unique_items_set: list of all unique item ids found from given dataset
    :return: the ratings matrix
    """

    # Create a map of unique user/item ids and their indices by which they will be stored in the matrix:
    users_vs_indices_map = {user: index for index, user in enumerate(unique_users_set)}
    items_vs_indices_map = {item: index for index, item in enumerate(unique_items_set)}

    # Initially, fill the matrix with NaNs:
    rating_matrix = np.full((len(unique_users_set), len(unique_items_set)), np.nan)

    # Fill the matrix with given ratings:
    for row in dataset:
        rating_matrix[users_vs_indices_map[row[0]]][items_vs_indices_map[row[1]]] = row[2]

    return rating_matrix, users_vs_indices_map, items_vs_indices_map


def initialize_user_matrix(num_unique_users, num_latent_factors):
    """
    Function that initializes the user matrix with random values that follow a Normal distribution with mean of 0 and
    standard deviation of 0.1, meaning that the generated values will be within +/- 0.1 of the mean (which is 0)
    :param num_unique_users: total number of unique user ids found in both the train and test sets
    :param num_latent_factors: number of latent factors (chosen after hyperparameter tuning)
    :return: the user matrix
    """
    np.random.seed(1)
    users_matrix = np.random.normal(0, 0.1, (num_unique_users, num_latent_factors))
    return users_matrix


def initialize_item_matrix(num_unique_items, num_latent_factors):
    """
    Function that initializes the item matrix with random values that follow a Normal distribution with mean of 0 and
    standard deviation of 0.1, meaning that the generated values will be within +/- 0.1 of the mean (which is 0)
    :param num_unique_items: total number of unique item ids found in both the train and test sets
    :param num_latent_factors: number of latent factors (chosen after hyperparameter tuning)
    :return: the item matrix
    """
    np.random.seed(1)
    items_matrix = np.random.normal(0, 0.1, (num_unique_items, num_latent_factors))
    return items_matrix


def run_ALS(num_iterations, users_items_matrix, users_matrix, items_matrix, regularization_term, num_latent_factors):
    """
    Function that runs the iterative alternating ALS process
    :param num_iterations: how many times to repeat the alternating ALS process
    :param users_items_matrix: matrix that stores ratings read from the training set
    :param users_matrix: user matrix initialized with random values
    :param items_matrix: item matrix initialized with random values
    :param regularization_term: regularization term (chosen after hyperparameter tuning)
    :param num_latent_factors: number of latent factors (chosen after hyperparameter tuning)
    :return: optimized user matrix and optimized item matrix, that will further be used for making predictions
    """

    # Iterate the alternating ALS process for num_iterations times:
    for i in range(0, num_iterations):

        # Item matrix I fixed, optimizing user matrix U:
        for user in range(users_matrix.shape[0]):
            # Get all items rated by this current user:
            items_rated_by_user = ~np.isnan(users_items_matrix[user, :])
            # Get the latent factors of all items rated by this current user:
            item_factors = items_matrix[items_rated_by_user, :]
            # Get the ratings that the current user has given to the items:
            ratings_by_user_for_item = users_items_matrix[user, items_rated_by_user]
            # Optimize the user matrix where it corresponds to the current user, while minimizing the loss function:
            users_matrix[user] = solve_least_squares(regularization_term, item_factors, ratings_by_user_for_item,
                                                     num_latent_factors)

        # User matrix U fixed, optimizing item matrix I:
        for item in range(items_matrix.shape[0]):
            # Get all users who rated this current item:
            user_who_rated_item = ~np.isnan(users_items_matrix[:, item])
            # Get the latent factors of all users who rated this current item:
            user_factors = users_matrix[user_who_rated_item, :]
            # Get the ratings that the current item was given by the users:
            ratings_for_item_by_user = users_items_matrix[user_who_rated_item, item]
            # Optimize the item matrix where it corresponds to the current item, while minimizing the loss function:
            items_matrix[item] = solve_least_squares(regularization_term, user_factors, ratings_for_item_by_user,
                                                     num_latent_factors)

        # TODO: remove
        print(i)
    return users_matrix, items_matrix


def solve_least_squares(regularization_term, factors, ratings_by_user_for_item, num_latent_factors):
    """
    Function that optimizes a user or item matrix by minimizing the loss function, while also using regularization
    :param regularization_term: the regularization term (chosen after hyperparameter tuning)
    :param factors: Get the latent factors of all items rated by a given user / all users who rated a given item
    :param ratings_by_user_for_item: ratings that the given user has given to items / the given item was given by users
    :param num_latent_factors: number of latent factors (chosen after hyperparameter tuning)
    :return: optimized user/item matrix
    """
    factors_transposed = factors.T

    # Calculate the regularization summand: \lambda * (identity matrix of size num_latent_factors * num_latent_factors)
    regularization = regularization_term * np.eye(num_latent_factors)

    # Apply the optimization formula
    # For optimizing the user matrix: U_u = (I_u^T I_u + \lambda I)^{-1} I_u^T R_u
    # For optimizing the item matrix: I_i = (U_i^T U_i + \lambda I)^{-1} U_i^T R_i
    return np.linalg.solve(factors_transposed @ factors + regularization, factors_transposed @ ratings_by_user_for_item)


def predict_average_when_cannot_be_predicted(item_index, user_index, item_average_ratings, user_average_ratings):
    """
        Function that is used when rating cannot be predicted directly. For instance, in cold start problem.
        :param user_index: index of the user, for whom the rating is being predicted
        :param item_index: index of the item, for which the rating is being predicted
        :param item_average_ratings: mean ratings of each item
        :param user_average_ratings: mean ratings of each user
        :return: the item's or user's average rating
    """

    # If the item has been rated by some users, return the item's average rating:
    mean_item_rating = item_average_ratings[item_index]
    if mean_item_rating != 0:
        return mean_item_rating

    # If the item hasn't been rated yet, return the user's average rating:
    mean_user_rating = user_average_ratings[user_index]
    return mean_user_rating


def get_average_ratings(users_items_matrix, for_users):
    """
        Function that calculates average ratings of each user / item
        :param users_items_matrix: matrix, where rows = user ids, columns = item ids, and values = ratings
        :param for_users: if True, this function will return average rating of each user. Otherwise, it will
                          return average ratings of each item.
        :return: list of average ratings of each user / item
    """
    if for_users:
        ax = 1
    else:
        ax = 0

    # Only include non-zero ratings in calculation of means:
    non_zero_ratings = (users_items_matrix != 0)

    # Calculate the mean user/item ratings:
    non_zero_ratings_sum = np.sum(users_items_matrix * non_zero_ratings, axis=ax)
    non_zero_ratings_count = np.sum(non_zero_ratings, axis=ax)
    non_zero_ratings_count[non_zero_ratings_count == 0] = 1
    mean_ratings = non_zero_ratings_sum / non_zero_ratings_count
    return mean_ratings


def predict_ratings(testing_set, users_matrix, items_matrix, users_and_indices, items_and_indices, users_vs_items_matrix):
    """
    Function that predicts ratings for the testing set and writes them to results.csv file
    :param testing_set: the testing set for which to predict ratings
    :param users_matrix: ALS-optimized user matrix
    :param items_matrix: ALS-optimized item matrix
    :param users_and_indices: map of user ids and their indices by which they are stored in matrices
    :param items_and_indices: map of item ids and their indices by which they are stored in matrices
    :param users_vs_items_matrix: matrix, where rows = user ids, columns = item ids, and values = given ratings from
    the training set
    """

    # Calculate and store average ratings of each item and user, to later use in such cases like cold start problem:
    item_average_ratings = get_average_ratings(users_vs_items_matrix, False)
    user_average_ratings = get_average_ratings(users_vs_items_matrix, True)

    # TODO: remove
    predictions = []
    counter = 0

    # Open the file for writing:
    with open('results.csv', 'w') as file:
        # Iterate over each item of the testing set:
        for user, item, _, timestamp in testing_set:
            # If a rating can be predicted:
            if user in users_and_indices and item in items_and_indices:
                user_index = users_and_indices[user]
                item_index = items_and_indices[item]
                # Predict the rating by dot-multiplying the optimized user and item matrices:
                rating = np.dot(users_matrix[user_index], items_matrix[item_index])

                # TODO: remove
                predictions.append([user, item, rating])
            # If a rating cannot be predicted (i.e. cold start problem / user or item have not been in training set:
            else:
                # TODO: remove
                predictions.append([user, item, np.nan])
                counter += 1

                # Predict the rating to be the item's average rating (if the item has been rated by some users) or the
                # user's average rating (if the item hasn't been rated yet)
                rating = predict_average_when_cannot_be_predicted(item_index, user_index, item_average_ratings,
                                                                  user_average_ratings)

            # Write the rating to results.csv file:
            file.write(f"{user},{item},{round(rating)},{timestamp}\n")

    # TODO: remove
    print("NaN:", counter)
    return np.array(predictions)


# TODO: move this to after __main__
"""
The following 4 functions are not used in the final code.
However, they were used for the hyperparameter tuning process. 
"""


def train_validation_split(data, validation_size, random_seed):
    """
    Function for splitting the given training set into training and validation sets that are used for hyperparameter
    tuning
    :param data: the training dataset
    :param validation_size: size of the validation set (in per cents)
    :param random_seed: random seed used for re-trainings
    :return: training set and validation set
    """
    np.random.seed(random_seed)
    # Shuffle the data to make it randomly ordered:
    shuffled_indices = np.random.permutation(len(data))
    # Size of the resulting validation set:
    validation_set_size = int(len(data) * validation_size)
    # Split the data into training set and validation set, and return them:
    return data[shuffled_indices[validation_set_size:]], data[shuffled_indices[:validation_set_size]]


def calculate_mae(predictions, actuals):
    """
    Function for calculating the MAE score of predictions
    :param predictions: predicted ratings
    :param actuals: actual ratings
    :return: the MAE score
    """
    return np.mean(np.abs(predictions - actuals))


def evaluate_hyperparameters(num_iterations, num_latent_factors, regularization_term, users_vs_items_matrix,
                             users_and_indices, items_and_indices, testing_set):
    """
    Function for computing the MAE score of one set of hyperparameters
    :param num_iterations: hyperparameter - how many times to repeat the alternating ALS process
    :param num_latent_factors: hyperparameter - number of latent factors
    :param regularization_term: hyperparameter - regularization term
    :param users_vs_items_matrix: matrix, where rows = user ids, columns = item ids, and values = given ratings from
    the training set
    :param users_and_indices: map of user ids and their indices by which they are stored in matrices
    :param items_and_indices: map of item ids and their indices by which they are stored in matrices
    :param testing_set: the testing set
    :return: MAE score for this set of hyperparameters
    """
    users_matrix = initialize_user_matrix(len(unique_users), num_latent_factors)
    items_matrix = initialize_item_matrix(len(unique_items), num_latent_factors)

    users_matrix_after_ALS, items_matrix_after_ALS = run_ALS(num_iterations, users_vs_items_matrix, users_matrix,
                                                             items_matrix, regularization_term, num_latent_factors)

    predicted_ratings = predict_ratings(testing_set, users_matrix_after_ALS, items_matrix_after_ALS, users_and_indices,
                                        items_and_indices, users_vs_items_matrix)
    actual_ratings = testing_set[:, :-1]

    mae_score = calculate_mae(np.array(predicted_ratings), actual_ratings)
    return mae_score


def grid_search_hyperparameter_tuning(users_vs_items_matrix, users_and_indices, items_and_indices, testing_set):
    """

    :param users_vs_items_matrix: matrix, where rows = user ids, columns = item ids, and values = given ratings from
    the training set
    :param users_and_indices: map of user ids and their indices by which they are stored in matrices
    :param items_and_indices: map of item ids and their indices by which they are stored in matrices
    :param testing_set: the testing set
    :return: the best-performing set of hyperparameters (that yields the lowest MAE score)
    """
    num_iterations_grid = [7, 10] # 2, 5, 10, 15
    num_latent_factors_grid = [7, 10] # 5, 7, 10, 20, 30
    regularization_term_grid = [0.1, 0.2] # 0.01, 0.05, 0.1, 0.15, 0.2

    best_mae = float('inf')
    best_params = None

    # For each configuration of hyperparameters:
    for num_iterations in num_iterations_grid:
        for num_latent_factors in num_latent_factors_grid:
            for regularization_term in regularization_term_grid:
                # Compute the MAE score of this set of hyperparameters:
                mae = evaluate_hyperparameters(num_iterations, num_latent_factors, regularization_term,
                                               users_vs_items_matrix, users_and_indices, items_and_indices, testing_set)
                print(num_iterations, num_latent_factors, regularization_term, f"MAE = {mae}")

                # Check if this MAE is better than all previous ones:
                if mae < best_mae:
                    best_mae = mae
                    best_params = (num_iterations, num_latent_factors, regularization_term)

    # Return the best set of hyperparameters and the acquired MAE score:
    return best_params, best_mae




if __name__ == '__main__':
    # TODO: remove
    start_time = time.time()

    # Read the train set and test set from given files into numpy arrays:
    train_set = read_dataset_from_file("train_20M_withratings.csv")
    # test_set = read_dataset_from_file("test_20M_withoutratings.csv")

    # TODO: remove
    # train_set = read_dataset_from_file("train_100k_withratings.csv")
    # test_set = read_dataset_from_file("test_100k_withoutratings.csv")

    # TODO: remove
    train_set, test_set = train_validation_split(train_set, 0.1, 42)

    # Get the lists of unique users and items given in files:
    unique_users, unique_items = get_unique_users_items(train_set, test_set)
    print("Unique:", len(unique_users), len(unique_items))

    # Store the given ratings in matrix form:
    ratings_matrix, users_vs_indices, items_vs_indices = create_ratings_matrix(
        train_set, unique_users, unique_items)
    print("Given ratings:", ratings_matrix.shape, len(users_vs_indices), len(items_vs_indices))

    # TODO: uncomment
    # Initialize the user matrix and the item matrix with random values:
    # user_matrix = initialize_user_matrix(len(unique_users), 20)
    # item_matrix = initialize_item_matrix(len(unique_items), 20)

    # TODO: remove
    # user_matrix_old = np.copy(user_matrix)
    # item_matrix_old = np.copy(item_matrix)

    # # Store the given timestamps in matrix form:
    # timestamps_matrix = create_timestamps_matrix(train_set, test_set, users_vs_indices, items_vs_indices)

    # TODO: uncomment
    # Run the iterative alternating ALS process with the hyperparameters identified by hyperparameter tuning:
    # user_matrix_after_ALS, item_matrix_after_ALS = run_ALS(2, ratings_matrix, user_matrix, item_matrix,
    #                                                        0.01, 20)

    # TODO: remove
    # identical = np.array_equal(user_matrix_old, user_matrix_after_ALS)
    # if identical:
    #     print("User arrays are completely identical.")
    # else:
    #     print("User arrays are not completely identical.")
    # identical2 = np.array_equal(item_matrix_old, item_matrix_after_ALS)
    # if identical2:
    #     print("Item arrays are completely identical.")
    # else:
    #     print("Item arrays are not completely identical.")

    # Predict the ratings for the testing set and write them to results.csv file:
    # TODO: method should not return anything eventually
    # predicted_ratings = predict_ratings(test_set, user_matrix_after_ALS, item_matrix_after_ALS, users_vs_indices,
    #                                     items_vs_indices, ratings_matrix)

    # TODO: remove
    # actual_ratings = test_set[:, :-1]
    # print(predicted_ratings)
    # print(actual_ratings)
    # mae_score = calculate_mae(np.array(predicted_ratings), actual_ratings)
    # print(f"MAE Score: {mae_score}")

    # TODO: remove
    grid_search_hyperparameter_tuning(ratings_matrix, users_vs_indices, items_vs_indices, test_set)

    # TODO: remove
    end_time = time.time()
    total_time = end_time - start_time
    print("Elapsed time:", total_time, "seconds")
    os.system("say 'Your Python script has finished running'")
