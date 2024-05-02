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
         2.3. Latent factors = 
"""


def read_dataset_from_file(filename):
    """
    Function for reading a dataset from given file and storing it as a numpy array
    :param filename: the name of the file to read
    :return: numpy array with all contents of the file stored inside
    """
    dataset = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        # Read the file line by line:
        for line in reader:
            # Split each line on commas and store it in array:
            row = [int(item) if item.isdigit() else float(item) if '.' in item else item for item in line[0].split(",")]
            dataset.append(row)
    return np.array(dataset, dtype=object)


def get_unique_users_items(dataset1, dataset2):
    """
        Function that finds all unique users and all unique items within both the train set and test set,
        and returns them stored in 2 lists
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


def initialize_user_matrix(num_unique_users, num_latent_factors):
    """
    Function that initializes the user matrix with random values that follow a Normal distribution with mean of 0 and
    standard deviation of 0.1, meaning that the generated values will be within +/- 0.1 of the mean (which is 0)
    :param num_unique_users:
    :param num_latent_factors:
    :return:
    """
    np.random.seed(1)
    users_matrix = np.random.normal(0, 0.1, (num_unique_users, num_latent_factors))
    return users_matrix


def initialize_item_matrix(num_unique_items, num_latent_factors):
    # np.random.seed(1)
    items_matrix = np.random.normal(0, 0.1, (num_unique_items, num_latent_factors))
    return items_matrix


def create_ratings_matrix(dataset, unique_users_set, unique_items_set):
    """
    Function that creates a matrix, where rows are user ids, columns are item ids, and values are the ratings
    :param dataset: data read from one of the given files
    :param unique_users_set: list of all unique user ids found from given dataset
    :param unique_items_set: list of all unique item ids found from given dataset
    :return: the given ratings matrix
    """

    # Create a map of unique user/item ids and their indices by which they will be stored in the matrix:
    users_vs_indices_map = {user: index for index, user in enumerate(unique_users_set)}
    items_vs_indices_map = {item: index for index, item in enumerate(unique_items_set)}

    # Initially, fill the matrix with zeros:
    rating_matrix = np.full((len(unique_users_set), len(unique_items_set)), np.nan)

    # Fill the matrix with given ratings:
    for row in dataset:
        rating_matrix[users_vs_indices_map[row[0]]][items_vs_indices_map[row[1]]] = row[2]

    return rating_matrix, users_vs_indices_map, items_vs_indices_map


def run_ALS(num_iterations, users_items_matrix, users_matrix, items_matrix, regularization_term, num_latent_factors):
    for i in range(0, num_iterations):

        for user in range(users_matrix.shape[0]):
            items_rated_by_user = ~np.isnan(users_items_matrix[user, :])
            item_factors = items_matrix[items_rated_by_user, :]
            ratings_by_user_for_item = users_items_matrix[user, items_rated_by_user]
            users_matrix[user] = solve_least_squares(regularization_term, item_factors, ratings_by_user_for_item,
                                                     num_latent_factors)

        for item in range(items_matrix.shape[0]):
            user_who_rated_item = ~np.isnan(users_items_matrix[:, item])
            user_factors = users_matrix[user_who_rated_item, :]
            ratings_for_item_by_user = users_items_matrix[user_who_rated_item, item]
            items_matrix[item] = solve_least_squares(regularization_term, user_factors, ratings_for_item_by_user,
                                                     num_latent_factors)

        print(i)
    return users_matrix, items_matrix


def solve_least_squares(regularization_term, factors, ratings_by_user_for_item, num_latent_factors):
    factors_transposed = factors.T
    regularization = regularization_term * np.eye(num_latent_factors)
    return np.linalg.solve(factors_transposed @ factors + regularization, factors_transposed @ ratings_by_user_for_item)


def predict_average_when_cannot_be_predicted(item_index, user_index, item_average_ratings, user_average_ratings):
    """
        Function that is used when rating cannot be predicted directly. For instance, in cold start problem.
        :param user_index: index of the user, for whom the rating is being predicted
        :param item_index: index of the item, for which the rating is being predicted
        :param item_average_ratings: mean ratings of each item
        :param user_average_ratings: mean ratings of each user
        :return: the item's or user's average rating:
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
    item_average_ratings = get_average_ratings(users_vs_items_matrix, False)
    user_average_ratings = get_average_ratings(users_vs_items_matrix, True)

    predictions = []
    counter = 0
    with open('results.csv', 'w') as file:
        for user, item, timestamp in testing_set:
            if user in users_and_indices and item in items_and_indices:
                user_index = users_and_indices[user]
                item_index = items_and_indices[item]
                rating = np.dot(users_matrix[user_index], items_matrix[item_index])
                predictions.append([user, item, rating])
            else:
                # Handle case where user or item might not have been in the training set
                # TODO: remove line below and re-phrase comment above
                predictions.append([user, item, np.nan])

                rating = predict_average_when_cannot_be_predicted(item_index, user_index, item_average_ratings,
                                                                  user_average_ratings)
                counter += 1
            file.write(f"{user},{item},{round(rating)},{timestamp}\n")
    print("NaN:", counter)
    return np.array(predictions)


# TODO: remove
def train_test_split(data, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    # Shuffle the data
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]


# TODO: remove
def calculate_mae(predictions, actuals):
    return np.mean(np.abs(predictions - actuals))


if __name__ == '__main__':
    # TODO: remove
    start_time = time.time()

    # Read the train set and test set from given files into numpy arrays:
    # train_set = read_dataset_from_file("train_100k_withratings.csv")
    # test_set = read_dataset_from_file("test_100k_withoutratings.csv")
    train_set = read_dataset_from_file("train_20M_withratings.csv")
    test_set = read_dataset_from_file("test_20M_withoutratings.csv")

    # TODO: remove
    # train_set, test_set = train_test_split(train_set, test_size=0.1, random_seed=42)

    # Get the lists of unique users and items given in files:
    unique_users, unique_items = get_unique_users_items(train_set, test_set)
    print("Unique:", len(unique_users), len(unique_items))

    # Initialize the user matrix and the item matrix with random values:
    user_matrix = initialize_user_matrix(len(unique_users), 20)
    item_matrix = initialize_item_matrix(len(unique_items), 20)
    print("Initialized matrices:", user_matrix.shape, item_matrix.shape)

    # TODO: remove
    user_matrix_old = np.copy(user_matrix)
    item_matrix_old = np.copy(item_matrix)

    # Store the given ratings in matrix form:
    ratings_matrix, users_vs_indices, items_vs_indices = create_ratings_matrix(
        train_set, unique_users, unique_items)
    print("Given ratings:", ratings_matrix.shape, len(users_vs_indices), len(items_vs_indices))

    # # Store the given timestamps in matrix form:
    # timestamps_matrix = create_timestamps_matrix(train_set, test_set, users_vs_indices, items_vs_indices)

    user_matrix_after_ALS, item_matrix_after_ALS = run_ALS(2, ratings_matrix, user_matrix, item_matrix,
                                                           0.01, 20)

    # TODO: remove
    identical = np.array_equal(user_matrix_old, user_matrix_after_ALS)
    if identical:
        print("User arrays are completely identical.")
    else:
        print("User arrays are not completely identical.")
    identical2 = np.array_equal(item_matrix_old, item_matrix_after_ALS)
    if identical2:
        print("Item arrays are completely identical.")
    else:
        print("Item arrays are not completely identical.")

    predicted_ratings = predict_ratings(test_set, user_matrix_after_ALS, item_matrix_after_ALS, users_vs_indices,
                                        items_vs_indices, ratings_matrix)
    # actual_ratings = test_set[:, :-1]
    #
    # print(predicted_ratings)
    # print(actual_ratings)
    #
    # mae_score = calculate_mae(np.array(predicted_ratings), actual_ratings)
    # print(f"MAE Score: {mae_score}")

    # TODO: remove
    end_time = time.time()
    total_time = end_time - start_time
    print("Elapsed time:", total_time, "seconds")
    os.system("say 'Your Python script has finished running'")
