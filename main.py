import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix


# load training data from file
def load_train_data(train_path):
    with open(train_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split('\t')
        sentiment = int(parts[0])
        review = parts[1]
        data.append((sentiment, review))
    return data


def preprocess_data(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    text = text.replace('#EOF', '')
    # Remove unwanted characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text


# load test data
def load_test_data(test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    return data


# convert dataset reviews to bag-of-words representation
def create_sparse_bow_vector(review, _word_to_index):
    bow_vector = lil_matrix((1, len(_word_to_index)), dtype=np.float32)
    _words = review.split()
    for word in _words:
        if word in _word_to_index:
            bow_vector[0, _word_to_index[word]] = 1
    return bow_vector


def calculate_precision(y_true, y_pred, positive_label=1):
    true_positives = np.sum((y_true == positive_label) & (y_pred == positive_label))
    predicted_positives = np.sum(y_pred == positive_label)

    precision = true_positives / (predicted_positives + 1e-15)
    return precision


def calculate_recall(y_true, y_pred, positive_label=1):
    true_positives = np.sum((y_true == positive_label) & (y_pred == positive_label))
    actual_positives = np.sum(y_true == positive_label)

    recall = true_positives / (actual_positives + 1e-15)
    return recall


def f1_score(y_true, y_pred):
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def principal_component_analysis(X, num_components):
    # Center the data
    mean = np.mean(X, axis=0)
    centered_data = X - mean

    # calculate the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)

    # calculate eigenvectors & eigen values
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Sort eigen vectors by eigen values
    indices = np.argsort(-eigen_values)
    sorted_eigenvectors = eigen_vectors[:, indices]

    # project the data to a new feature space
    projected_data = centered_data @ sorted_eigenvectors[:, num_components]

    return projected_data, sorted_eigenvectors, mean


# calculates the Euclidean distance between two vectors.
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifierSparse:
    def __init__(self, k):
        self.k = k

    def predict(self, X_train, y_train, X_test):
        predictions = []
        for test_instance in X_test:
            distances = [euclidean_distance(test_instance, x) for x in X_train]
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [y_train[k] for k in k_nearest_indices]

            # Handle the case where k_nearest_labels is empty
            if len(k_nearest_labels) == 0:
                predictions.append(-1)
            else:
                # Break ties by choosing the label with the smallest Euclidean distance
                min_distance_label = None
                min_distance = float('inf')
                for label in set(k_nearest_labels):
                    distance_for_label = np.sum([distances[i] for i, k in enumerate(k_nearest_labels) if k == label])
                    if distance_for_label < min_distance:
                        min_distance = distance_for_label
                        min_distance_label = label

                predictions.append(min_distance_label)
        return predictions

    def cross_validation(self, X_train_sparse, y_train, k_values, n_splits=5):
        accuracies = {}

        """
        Convert sparse matrix to dense numpy array
        for easier slicing &
        runtime improvement
        """

        X_train = X_train_sparse.toarray()
        fold_size = len(X_train) // n_splits

        for k in k_values:
            fold_accuracies = []

            for i in range(n_splits):
                # Split the data into training and validation sets
                val_start = i * fold_size
                val_end = (i + 1) * fold_size

                X_val = X_train[val_start:val_end]
                y_val = y_train[val_start:val_end]

                X_train_fold = np.concatenate((X_train[:val_start], X_train[val_end:]), axis=0)
                y_train_fold = np.concatenate((y_train[:val_start], y_train[val_end:]), axis=0)

                y_pred = self.predict(X_train_fold, y_train_fold, X_val)

                accuracy = np.mean(y_pred == y_val)
                fold_accuracies.append(accuracy)

            accuracies[k] = np.mean(fold_accuracies)

        return accuracies


if __name__ == "__main__":
    train_data = load_train_data("train.txt")
    labels, reviews = zip(*train_data)

    preprocessed_reviews = [preprocess_data(review) for review in reviews]

    # build vocabulary based on the training data
    vocab = Counter()
    for review in preprocessed_reviews:
        words = review.split()
        vocab.update(words)
    # Create mapping from word to index
    word_to_index = {word: idx for idx, (word, _) in enumerate(vocab.items())}

    # Convert training reviews to bag of words representation using scipy.sparse
    X_train_sparse = lil_matrix((len(preprocessed_reviews), len(word_to_index)), dtype=np.uint8)
    for i, review in enumerate(preprocessed_reviews):
        X_train_sparse[i] = create_sparse_bow_vector(review, word_to_index)

    y_train = np.array(labels)

    test_data = load_test_data("test.txt")

    preprocess_test_data = [preprocess_data(review) for review in test_data]

    # Convert test reviews to bag of words representation using scipy.sparse
    X_test_sparse = lil_matrix((len(preprocess_test_data), len(word_to_index)), dtype=np.float32)
    for i, review in enumerate(preprocess_test_data):
        X_test_sparse[i] = create_sparse_bow_vector(review, word_to_index)

    k_val = [3, 9, 20, 30, 50]
    knn_classifier = KNNClassifierSparse(k=5)

    accuracy_dict = knn_classifier.cross_validation(X_train_sparse, y_train, k_val, n_splits=10)
    for n, accuracy in accuracy_dict.items():
        print(f"Mean accuracy for k={n}: {accuracy}")

    k_values = list(accuracy_dict.keys())
    mean_accuracies = list(accuracy_dict.values())

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, mean_accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy vs. Neighbors (k) for knn classifier')
    plt.grid(True)
    plt.show()
