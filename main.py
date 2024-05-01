import numpy as np
import csv


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

print(read_dataset_from_file("train_20M_withratings.csv"))

if __name__ == '__main__':
    # Read the train set and test set from given files into numpy arrays:
    train_set = read_dataset_from_file("train_20M_withratings.csv")
    test_set = read_dataset_from_file("test_20M_withoutratings.csv")

    print(train_set)
    print(test_set)

    # # Get the lists of unique users and items given in files:
    # unique_users, unique_items = get_unique_users_items(train_set, test_set)
    #
    # # Store the given ratings in matrix form:
    # ratings_matrix, users_vs_indices, items_vs_indices = create_ratings_matrix(
    #     train_set, unique_users, unique_items)
    # # Store the given timestamps in matrix form:
    # timestamps_matrix = create_timestamps_matrix(train_set, test_set, users_vs_indices, items_vs_indices)
    #
    # # Calculate the adjusted cosine similarity matrix for each pair of items:
    # items_similarity_matrix = create_item_similarity_matrix(ratings_matrix)
    #
    # # Determine the neighbourhood for each item:
    # items_neighbourhoods = get_neighbourhood(items_similarity_matrix)
    #
    # # Predict ratings for user-item pairs given in the test set, and write them to the results.csv file:
    # predict_ratings(test_set, users_vs_indices, items_vs_indices, items_similarity_matrix, items_neighbourhoods,
    #                 ratings_matrix)
