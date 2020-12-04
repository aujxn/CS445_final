import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

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

print("Training classifiers for " + str(num_tags) + "most common tags\n")

# Train and test classifiers for each of the tags
print('{:30} {:30} {:10}'.format("Tag:", "Number of recipes with tag:", "Accuracy:"))
for tag, test_label, train_label in zip(most_common_tags, test_labels, train_labels):
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(train_samples, train_label)
    print('{:30} {:30} {:.2f}%'.format(
        tags[str(tag[0])], 
        str(tag[1]), 
        clf.score(test_samples, test_label)*100)
        )
