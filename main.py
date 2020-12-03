import json

with open("recipes.json", "r") as file:
    recipes = file.read()

with open("ingredients.json", "r") as file:
    ingredients = file.read()

recipes = json.loads(recipes)
ingredients = json.loads(ingredients)

print(recipes)
