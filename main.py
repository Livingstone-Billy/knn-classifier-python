import re
import numpy as np
from collections import Counter
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA


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
    # Remove unwanted characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text


# load test data
def load_test_data(test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip() for line in lines]
    return data


# convert reviews to bag-of-words representation
def create_sparse_bow_vector(review, _word_to_index):
    bow_vector = lil_matrix((1, len(_word_to_index)), dtype=np.float32)
    _words = review.split()
    for word in _words:
        if word in _word_to_index:
            bow_vector[0, _word_to_index[word]] = 1
    return bow_vector


# calculates the Euclidean distance between two vectors.
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1 - x2) ** 2)


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

    def cross_validation(self, X_train_sparse, y_train, k_values, n_splits=10, metric='accuracy'):
        accuracies = {}
        # Initialize KFold for data splitting
        kf = KFold(n_splits=n_splits)

        for k in k_values:
            fold_metrics = []

            for train_idx, val_idx in kf.split(X_train_sparse):
                X_train_fold, X_val = X_train_sparse[train_idx], X_train_sparse[val_idx]
                y_train_fold, y_val = y_train[train_idx], y_train[val_idx]

                y_pred = self.predict(X_train_fold, y_train_fold, X_val)

                if metric == 'precision':
                    fold_metrics.append(precision_score(y_val, y_pred, average='weighted'))
                elif metric == 'recall':
                    fold_metrics.append(recall_score(y_val, y_pred, average='weighted'))
                elif metric == 'f1':
                    fold_metrics.append(f1_score(y_val, y_pred, average='weighted'))
                # Add more metrics if needed

            if fold_metrics:
                accuracies[k] = np.mean(fold_metrics)

        return accuracies


if __name__ == "__main__":
    train_data = load_train_data("train1.txt")
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
    X_train_sparse = lil_matrix((len(preprocessed_reviews), len(word_to_index)), dtype=np.float32)
    for i, review in enumerate(preprocessed_reviews):
        X_train_sparse[i] = create_sparse_bow_vector(review, word_to_index)

    y_train = np.array(labels)

    test_data = load_test_data("test.txt")

    preprocess_test_data = [preprocess_data(review) for review in test_data]

    # Convert test reviews to bag of words representation using scipy.sparse
    X_test_sparse = lil_matrix((len(preprocess_test_data), len(word_to_index)), dtype=np.float32)
    for i, review in enumerate(preprocess_test_data):
        X_test_sparse[i] = create_sparse_bow_vector(review, word_to_index)

    k_val = [1, 3, 4, 5, 6, 9, 10]
    knn_classifier = KNNClassifierSparse(k=5)
    pca = PCA(n_components=100)  # Adjust the number of components as needed
    X_train_pca = pca.fit_transform(X_train_sparse.toarray())
    accuracy_dict = knn_classifier.cross_validation(X_train_pca, y_train, k_val, n_splits=20, metric='f1')
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
