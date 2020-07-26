import numpy as np


def generate_classifiers(features: np.ndarray):
    def add_classifier(idx: int, cr: int):
        dict_cf['if x' + str(idx + 1) + '>' + str(cr) + ' then 1 else -1'] \
            = (idx, np.vectorize(lambda x: 1 if x > cr else -1))
        dict_cf['if x' + str(idx + 1) + '<' + str(cr) + ' then 1 else -1'] \
            = (idx, np.vectorize(lambda x: 1 if x < cr else -1))

    dict_cf = {}
    for i in range(features.shape[1]):
        unique_elements = np.unique(X[:, i])
        criteria = unique_elements[0] - 1
        add_classifier(i, criteria)
        criteria = unique_elements[-1] + 1
        add_classifier(i, criteria)
        for j in range(unique_elements.size - 1):
            criteria = (unique_elements[j] + unique_elements[j + 1]) / 2
            add_classifier(i, criteria)
    return dict_cf


X = np.array([[3, 2050], [1, 2200], [2, 2090], [4, 2230], [5, 2330],
              [6, 2220], [6, 2390], [7, 2320], [8, 2330], [8, 2090]])

y = np.array([-1, 1, 1, -1, 1, -1, 1, 1, -1, -1])

w = np.array([0.1] * 10)

cf = generate_classifiers(X)

cf_result = {}
for (k, v) in cf.items():
    cf_result[k] = v[1](X[:, v[0]])

cf_compare = {}
for (k, v) in cf_result.items():
    comp = (v != y) + 0
    cf_compare[k] = comp

strong_classifier = []
remaining_samples = 10
count = 1
while remaining_samples != 0:
    cf_error = {}
    for (k, v) in cf_compare.items():
        cf_error[k] = v.dot(w)
    best_cf_desc = min(cf_error, key=cf_error.get)
    best_err = cf_error[best_cf_desc]
    best_cf = cf[best_cf_desc]
    best_cf_result = cf_result[best_cf_desc]
    print("Best classifier for iteration " + str(count) + ": " + best_cf_desc + ", error rate is " + str(best_err))
    coef = 0.5 * np.log((1.0 - best_err) / best_err)
    print("Coefficient for iteration " + str(count) + ": " + str(coef))
    before_norm = w * np.exp(-coef * best_cf_result * y)
    w = before_norm/np.sum(before_norm)
    print("New weight is: " + str(w))
    strong_classifier.append((coef, best_cf, best_cf_desc))
    count += 1
    new_result = np.zeros(y.shape)
    for i in strong_classifier:
        new_result += i[0]*i[1][1](X[:, i[1][0]])
    new_result = np.sign(new_result)
    remaining_samples = np.sum(new_result != y)
    print("Number of wrongly classified samples : " + str(remaining_samples))
print(str(strong_classifier))



