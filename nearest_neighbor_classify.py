import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k = 1):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    '''

    # 0) Gets the distance between each test image feature and each train image feature
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    # 1) Find the k closest features to each test image feature in euclidean space
    predictions = []

    for distance in distances:
        k_small_dis_labels = []
        sorted_dis_index = np.argsort(distance)

        # 2) Determine the labels of those k features
        for i in range(k):
            k_small_dis_labels.append(train_labels[sorted_dis_index[i]])
        
        # 3) Pick the most common label from the k
        most_common_label = Counter(k_small_dis_labels).most_common(1)[0][0]

        # 4) Store that label in a list
        predictions.append(most_common_label)

    return np.array(predictions)