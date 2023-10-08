import re
from collections import Counter

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


# convert dataset reviews to bag-of-words representation
def create_sparse_bow_vector(review, _word_to_index):
    bow_vector = lil_matrix((1, len(_word_to_index)), dtype=np.float32)
    _words = review.split()
    for word in _words:
        if word in _word_to_index:
            bow_vector[0, _word_to_index[word]] = 1
    return bow_vector


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.y_train = Y_train

    def predict(self, X_train, Y_train, X_test):
        predictions = []
        for test_inst in X_test:
            distances = [self.euclidean_distance(test_inst, x) for x in X_train]
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [Y_train[i] for i in k_nearest_indices]

            # Handle the case where k_nearest_labels is empty
            if len(k_nearest_labels) == 0:
                predictions.append(np.mod(Y_train)[0])
            else:
                predicted_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
                predictions.append(predicted_label)
        return predictions

    # Inside the KNNClassifier class
    def k_fold_cross_validation(self, X_train_sparse, y_train, k_values, n_splits=5):
        accuracies = {}

        # Convert sparse matrix to dense numpy array for easier slicing
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

                # Fit the classifier and predict
                self.fit(X_train_fold, y_train_fold)
                y_pred = self.predict(X_train_fold, y_train_fold, X_val)

                accuracy = np.mean(y_pred == y_val)
                fold_accuracies.append(accuracy)

            accuracies[k] = np.mean(fold_accuracies)

        return accuracies


def load_datasets(train_path, test_path):
    # Load train dataset
    with open(train_path, 'r', encoding='utf-8') as file:
        train_lines = file.readlines()

    train_data = []
    for line in train_lines:
        parts = line.strip().split('\t')
        sentiment = int(parts[0])
        review = parts[1]
        train_data.append((sentiment, review))

    # Load test dataset
    with open(test_path, 'r', encoding='utf-8') as file:
        test_lines = file.readlines()
    test_data = []
    for line in test_lines:
        parts = line.strip().split('\t')
        review = parts[0]
        test_data.append(review)

    return train_data, test_data


def preprocess_data(text):
    # Remove unnecessary html tags
    text = re.sub(r'<.*?>', '', text)
    # Remove #EOF token at end each review
    text = text.replace('#EOF', '')
    # Remove unwanted characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text


def preprocess_train_data(training_data_set):
    labels, reviews = zip(*training_data_set)
    preprocessed_reviews = [preprocess_data(review) for review in reviews]

    vocab = Counter()
    for review in preprocessed_reviews:
        words = review.split()
        vocab.update(words)

    word_to_idx = {word: index for index, (word, _) in enumerate(vocab.items())}

    X_train_sparse = lil_matrix((len(preprocessed_reviews), len(word_to_idx)), dtype=np.uint8)
    for i, review in enumerate(preprocessed_reviews):
        X_train_sparse[i] = create_sparse_bow_vector(review, word_to_idx)

    Y_train = np.array(labels)
    x_train_csr = csr_matrix(X_train_sparse)

    return x_train_csr, Y_train, word_to_idx


def preprocess_test_data(_testing_data, word_to_indices):
    preprocessed_reviews = [preprocess_data(review) for review in _testing_data]

    X_test_sparse = lil_matrix((len(preprocessed_reviews), len(word_to_indices)), dtype=np.uint8)
    for i, review in enumerate(preprocessed_reviews):
        X_test_sparse[i] = create_sparse_bow_vector(review, word_to_indices)
    x_test_csr = csr_matrix(X_test_sparse)

    return x_test_csr


if __name__ == "__main__":
    print("Model is now learning.....")
    k_test_values = [3, 5, 7, 9, 12]

    training_data, testing_data = load_datasets(train_path="train.txt", test_path="test.txt")

    # Preprocess train data
    X_train_csr, y_train, word_to_index = preprocess_train_data(training_data)

    # Preprocess test data
    X_test_csr = preprocess_test_data(testing_data, word_to_index)

    knn_classifier = KNNClassifier(k=5)
    knn_classifier.fit(X_train_csr, y_train)

    accuracy_dict = knn_classifier.k_fold_cross_validation(X_train_csr, y_train, k_test_values, n_splits=10)
    for n, accuracy in accuracy_dict.items():
        print(f"Mean accuracy for k={n}: {accuracy}")
