import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def score(predictions, labels): 
    true_neg = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for prediction, label in zip(predictions, labels):
        if prediction == 0 and label == 0:
            true_neg += 1
        elif prediction == 1 and label == 1:
            true_pos += 1
        elif prediction == 0 and label == 1:
            false_neg += 1
        else:
            false_pos += 1
    return {
            "precision": 100 * true_pos / (true_pos + false_pos),
            "recall": 100 * true_pos / (true_pos + false_neg),
            "accuracy": 100 * (true_pos + true_neg) / len(predictions)
           }

# Load the data
with open("recipes.json", "r") as file:
    recipes = file.read()
with open("ingredients.json", "r") as file:
    ingredients = file.read()
with open("tags.json", "r") as file:
    tags = file.read()
recipes = json.loads(recipes)
ingredients = json.loads(ingredients)
tags = json.loads(tags)

print("number of recipes: " + str(len(recipes)))

# Create a mapping to make the one hot encoding of ingredients with
ingredients_indices = {}
index = 0
for ingredient in ingredients.items():
    ingredients_indices[ingredient[0]] = index
    index += 1

# Count how often the tags show up in recipes
tag_counts = {}
for recipe in recipes.items():
    for tag in recipe[1]["tags"]:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 0
sorted_tags = sorted(tag_counts.items(), key=lambda item: item[1])

# Take the most common tags found
num_tags = 50
most_common_tags = sorted_tags[-num_tags:]

# Calculate the dimension of the training and testing data
num_samples = len(recipes)
num_features = len(ingredients)
# Currently using 80% train 20% test
train_size = (num_samples * 4) // 5

labels = np.zeros((num_samples, num_tags))
samples = np.zeros((num_samples, num_features))

# Create the training and testing data
sample_index = 0
for recipe in recipes.items():
    tag_index = 0
    for tag in most_common_tags:
        if tag[0] in recipe[1]["tags"]:
            labels[sample_index][tag_index] = True
        tag_index += 1

    for ingredient in recipe[1]["ingredients"]:
        samples[sample_index][ingredients_indices[str(ingredient["id"])]] = 1.0
    sample_index += 1

# Split the data into training and testing sets
train_samples = samples[:train_size]
test_samples = samples[train_size:]
train_labels = np.transpose(labels[:train_size])
test_labels = np.transpose(labels[train_size:])

print("Training classifiers for " + str(num_tags) + " most common tags\n")

# Train and test classifiers for each of the tags
print('{:30} {:15} {:25} {:15} {:15}'.format("Tag:", "# with tag:", "(log/bayes) Precision:", "Recall:", "Accuracy:"))
for tag, test_label, train_label in zip(most_common_tags, test_labels, train_labels):
    log_reg_clf = LogisticRegression(random_state=0, max_iter=1000)
    log_reg_clf.fit(train_samples, train_label)
    bayes_clf = GaussianNB()
    bayes_clf = bayes_clf.fit(train_samples, train_label)

    logistic_regression_predictions = log_reg_clf.predict(test_samples)
    naive_bayes_predictions = bayes_clf.predict(test_samples)

    logistic_regression_scores = score(logistic_regression_predictions, test_label)
    naive_bayes_scores = score(naive_bayes_predictions, test_label)

    print('{:30} {:15} {:25} {:15} {:15}'.format(
        tags[str(tag[0])], 
        str(tag[1]), 
        '{:05.2f}  {:05.2f}'.format(logistic_regression_scores['precision'], naive_bayes_scores['precision']),
        '{:05.2f}  {:05.2f}'.format(logistic_regression_scores['recall'], naive_bayes_scores['recall']),
        '{:05.2f}  {:05.2f}'.format(logistic_regression_scores['accuracy'], naive_bayes_scores['accuracy']),
        ))

