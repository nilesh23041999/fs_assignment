# fs_assignment

# Problem Statement: Finding Visually Similar Garments for an Input Garment

Garments in the fashion domain can be of multiple shapes, sizes, and colors. Finding garments similar to
each other is an important feature used by e-commerce websites to show recommendations to its users.
We would like to find visually similar garments for any input garment from within a given dataset of garment
images.

# Solution
A python project that outputs 'n' similar images from the database of an input query image. 
___

* First, we need all the requirements to be installed.
```python
pip install -r requirements.txt
```
___ 

* We need embeddings of the database to be compared against a query image. 
A python file *create_embeddings.py* is used to create embeddings.
```python
python create_embeddings.py
``` 
  This will create embeddings for us.
___

* Now to get 'n' similar images from the database, we will run main.py file
```python
python main.py --input "path of input image" --count "number of similar images to be querried from database" --embedding_path "path of embeddings"
```
___

* Future Work can be improving the feature extraction of the database. We can use other pretrained models and compare the results. We can also cluster the images rather than checking every image in the database to find the similar features.
We can also improve the feature comparison metric like comparing the existing results with euclidean distance etc. 

