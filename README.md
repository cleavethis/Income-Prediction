# Adult Income Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cleavethis/Income-Prediction/blob/main/ML_Income_Prediction.ipynb)

Predicting whether an individual's income exceeds $50K/year using the
UCI Adult (Census Income) dataset.

## Overview

The goal for this project is to build a binary classifier to determine if an
individual's income is above or below $50,000.

The dataset used is a US census dataset from 1994. The final result showed
that the most important feature for classification was a person's marital
status, specifically those married to civilians. Gradient boosting `feature_importances` reported
the magnitude, with marital status ranking much higher than any other feature.
Logistic regression signed coefficients reported the <em>direction</em>, `Married-civ-spouse` showing
it pushes strongly toward the >50K class, while `Never-married` pushes strongly the other way.

## Dataset

- Source: [UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- ~32,500 rows, 14 features, binary target (`>50K` / `<=50K`)
- Class distribution: ~75% <=50K, ~25% >50K

## Data Cleaning

The dataset used `"?"` for missing data values in 3 columns: `workclass`,
`occupation`, and `native-country`.

First, the missing rows were evaluated to determine how much overlap there
was between missing values, and 1,836 rows had overlap between `workclass`
and `occupation`. `native-country` missing rows contributed an additional
~500 missing entries.

For this project, rows with missing data were dropped. This represented a
small amount of data (~5%) and was a simpler solution compared to
alternatives such as imputing the data or keeping `"?"` as its own category.

## Feature Engineering

One-hot encoding was used for nominal categories. It may be counterintuitive
to have the model learn about a category like country of origin and derive
some meaning from them being ordered.

Sex is binary in this dataset, so it was mapped directly to 0/1 rather
than one-hot encoded, which would have just produced two redundant mirror-image
columns. For the target label `income`, `>50K` was assigned `1` and `<=50K`
was assigned `0`, as greater than 50 thousand represents the positive
classification.

Education was dropped, as the feature `education-num` is used to represent
highest education level — here the ordinal nature of further education could
be relevant to the model's learning.

## Modeling Approach

- The split ratio for data was 80% train, 20% test. This gives a robust
  collection of data to train on, with enough test material to draw some
  conclusions.
- Features were scaled as part of preprocessing, so that some features
  don't dominate others naturally when they have different ranges.
- The scaler was fit only on the training set, so no information from the test
  set (which is supposed to represent unseen, future data) leaked into 
  preprocessing.
- The first approach used was a simple logistic regression model, as this is
  the model I understand the most and wanted to see it in action.
- After reweighting alone (`class_weight="balanced"`) only traded precision
  for recall without meaningfully improving F1, I moved to tree-based
  models, since logistic regression can only learn a linear decision
  boundary and can't capture interactions between multiple features (e.g.
  `hours-per-week` likely matters differently depending on occupation). I
  started with Random Forest as a straightforward, less finicky-to-tune
  tree ensemble.

## Handling Class Imbalance


75% of the data is labeled as class 0 `<=50K` so the accuracy of the model is misleading as
guessing 0 for 100% of the entries would result in ~75% accuracy, so I turned to other measurements: precision, recall, and F1 score.

The LR model missed many of the >50k earners, so my first approach was to weight the `1` class higher, penalizing the missed examples
more harshly, hopefully catching more high earners.

Balancing the classes resulted in a significant gain in recall, but precision dropped in turn. Thus I decided that F1 score was 
the measure I wanted to improve, as I cared about both precision and recall. Precision is: when the model predicts >50K, how often is it actually right
Whereas recall measures: out of all 50K earners, how many did the model find? So precision and recall often affect each other inversely. As the model predicts
more 50K earners, it simultaneously mislabels more <50K earners incorrectly. 


## Results

| Model | Accuracy | Precision (>50K) | Recall (>50K) | F1 (>50K) |
|---|---|---|---|---|
| Logistic Regression (baseline) | .85|.76|.62|.68 |
| Logistic Regression (balanced) | .81|.59|.84|.69|
| Gradient Boosting | .82| .60| .86| .71|
| XGBoost | .84| .64| .86| .73|



From the results we can observe that XGBoost had the highest F1 score. This small gain was due to the ensemble learning approach. Both boosting models uses sequential
trees to correct the errors of the previous trees, resulting in more robust predictions. Recall for the positive class was tied between Gradient Boosting and the XGBoost model. 
However, the built in regularization of XGBoost avoids overfitting because overly complex trees are discouraged. It also allows for more hyperparameters (which is where this project would look next to further improve results).

## Feature Importance

| Feature | Importance |
|---|---|
| `marital-status_Married-civ-spouse` | 0.475 |
| `education-num` | 0.156 |
| `capital-gain` | 0.147 |
| `age` | 0.084 |
| `hours-per-week` | 0.043 |
| `capital-loss` | 0.033 |
| `occupation_Exec-managerial` | 0.011 |
| `occupation_Prof-specialty` | 0.007 |
| `occupation_Farming-fishing` | 0.007 |
| `occupation_Other-service` | 0.007 |
| `relationship_Wife` | 0.006 |
| `fnlwgt` | 0.003 |
| `sex` | 0.003 |
| `marital-status_Never-married` | 0.002 |
| `relationship_Own-child` | 0.002 |


The built in attribute `feature_importances` scores every feature based on how much it contributed to reducing error across all trees, averaged over the entire ensemble. Marital
status (civilian spouse) was 3 times more influential than the next closest feature, specifically those who are married tend to fall into the 1 class much more often than those who are unmarried or married to armed forces members. Since this dataset is reported as individual income, this is not simply a case of combined household income, though there is the possibility that people reported
incorrectly. I can only hypothesize other explanations such as age and career stage. Married people are likely to be older and more established in their careers, and a dual-income
household is likely more stable than a single-income household.

## What I'd Try Next

The obvious next steps are twofold. First, I'd like to use a SVC on the data and compare how it performs against these results. I think this is a good first step to get a new baseline
before moving into hyperparameter tuning. Which leads to the second step, hyperparameter searching. I would take the winner of this preliminary exploration XGBoost and SVC and search for which
hyperparameters would give the most gain here. 

