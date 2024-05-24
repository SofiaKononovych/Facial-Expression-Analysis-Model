import numpy as np
import csv

def import_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]

    #Separate numerical data from class names
    numerical_data = np.array([list(map(float, row[:-1])) for row in data])
    names_data = np.array([row[-1] for row in data])
    return numerical_data, names_data

def calc_mean_cov(data):
    mean_vec = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    return mean_vec, cov_matrix

#Calculate prior probability for class
def calc_p_ci(class_samples, total):
    return class_samples / total

def gdf(x, mean_vec, cov, prior_prob, n_features):
    x_mean = x - mean_vec
    cov_inv = np.linalg.inv(cov)
    normalization_res = -0.5 * n_features * np.log(2 * np.pi)
    log_det_cov = -0.5 * np.log(np.linalg.det(cov))
    exponent_res = -0.5 * np.dot(x_mean.T, np.dot(cov_inv, x_mean))
    gdf = normalization_res + log_det_cov + exponent_res + np.log(prior_prob)

    return gdf

def train_model(numerical_data, name_data):
    smile_data = numerical_data[name_data == 'smile']
    frown_data = numerical_data[name_data == 'frown']

    mean_smile, cov_smile = calc_mean_cov(smile_data)
    mean_frown, cov_frown = calc_mean_cov(frown_data)

    prior_smile = calc_p_ci(len(smile_data), len(numerical_data))
    prior_frown = calc_p_ci(len(frown_data), len(numerical_data))

    return mean_smile, cov_smile, prior_smile, mean_frown, cov_frown, prior_frown

def test_model(numerical_data, name_data, n_features, mean_1, cov_1, prior_1, mean_2, cov_2, prior_2 ):
    predictions = []

    for i in range(len(numerical_data)):
        sample = numerical_data[i]

        gdf_smile = gdf(sample, mean_1, cov_1, prior_1, n_features)
        gdf_frown = gdf(sample, mean_2, cov_2, prior_2, n_features)

        predicted = 'smile' if gdf_smile > gdf_frown else 'frown'
        predictions.append(predicted)

    return np.array(predictions)


#Calculate error rate, the percentage of times approach maps a vector into the wrong class
def evaluate_error(predictions, actual):
    error_rate = np.mean(predictions != actual)
    return error_rate

#Load training and test data
training_numerical, training_class_names = import_data("training-part-2 (1).csv")
test_numerical, test_class_names = import_data("test-part-2 (1).csv")

n_features = training_numerical.shape[1]

#Process of training the model
mean_smile, cov_smile, prior_smile, mean_frown, cov_frown, prior_frown \
    = train_model(training_numerical, training_class_names, n_features)

#Testing the model
predictions = \
    test_model(test_numerical, test_class_names, n_features, mean_smile, cov_smile, prior_smile,
               mean_frown, cov_frown, prior_frown)

error_rate = evaluate_error(predictions, test_class_names)

print("Results of training the model: error rate = ", error_rate)