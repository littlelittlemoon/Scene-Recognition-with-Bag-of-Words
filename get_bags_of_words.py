import numpy as np
from skimage.io import imread_collection
from skimage.feature import hog
from scipy.spatial.distance import cdist
from numpy.linalg import norm

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')

    images = imread_collection(image_paths)
    images_histograms = []

    cells_per_block = (2, 2) # Change for lower compute time
    t = cells_per_block[0]
    pixels_per_cell = (4, 4)
    images_feature_vectors = []

    for i, image in enumerate(images):
        feature_vector = hog(
            image, 
            feature_vector = True, 
            pixels_per_cell = pixels_per_cell,
            cells_per_block = cells_per_block, 
            visualize = False
        ).reshape(-1, t*t*9)

        # 计算当前图片的feature与词袋的距离
        dist = cdist(vocab, feature_vector, metric='euclidean')

        # 选择最短距离，计算直方图
        min_dis_index = np.argmin(dist, axis=0)
        histogram, bin_edges = np.histogram(min_dis_index, bins=len(vocab))
        histogram = histogram / norm(histogram)

        images_histograms.append(histogram)

    return np.array(images_histograms)