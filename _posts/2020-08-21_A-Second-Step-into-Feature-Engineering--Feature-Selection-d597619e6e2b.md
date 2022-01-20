---
title: 'A Second Step into Feature Engineering: Feature Selection'
date: 2020-08-21
permalink: /posts/2020/08/A-Second-Step-into-Feature-Engineering:Feature-Selection/
tags:
  - feature engineering
  - feature selection
---


> We are ready to start with the second part of Feature Engineering (if you’ve missed the previous article, you can find it [here](https://medium.com/mljcunito/an-introduction-to-feature-engineering-feature-importance-7e8265eb3a36)). In this short article, we’ll go through a few simple techniques in Feature Selection and Extraction.

![](img/1__XHHToil9E5EFeEh0H0rnjA.jpeg)

> Not all features are created equal

> Zhe Chen

### Feature Selection

There would always be some features that are less important with respect to a specific problem. Those irrelevant features need to be removed. _Feature selection_ addresses these problems by automatically selecting a subset that is most useful to the problem.

Most of the time the reduction in the number of input variables shrinks the computational cost of modeling, but sometimes it might happen that it also improves the performance of the model.

Among a large amount of feature selection methods, we’ll focus mainly on statistical-based ones. They involve evaluating the relationship between each input variable and the target variable using statistics. These methods are usually fast and effective, the only issue is that statistical measures depend on the data type of both input and output variables.

The classes in the `sklearn.feature_selection` module can be used for feature selection/dimensionality reduction on sample sets.

Whenever you want to go for a simple approach, there’s always a _threshold_ involved. `VarianceThreshold` is a simple baseline approach to select features. It removes all features whose variance doesn't reach a certain threshold.

```
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
#Load Data 
iris = datasets.load_iris()
# Create features and target 
X = iris.data
y = iris.target
# Conduct Variance Thresholding 
thresholder = VarianceThreshold(threshold = .6)
X_high_variance = thresholder.fit_transform(X)
# View High Variance Features 
print(X_high_variance[0:5])
print(X[0:5])
```

#### Univariate Feature Selection

Univariate feature selection examines each feature individually to determine the strength of the relationship of the feature with the response variable.

There are a few different options for univariate selection:

We can perform chi-squared (𝝌²) test on the samples to retrieve only the two best features:

```
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Load iris dataset
X, y = load_iris(return_X_y=True)
print(X.shape)
# retrieve the two best features by chi-squared test
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)
```

We have different scoring functions for regression and classification, some of them are listed here:

*   Regression: [f\_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression), [mutual\_info\_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression)
*   Classification: [chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2), [f\_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif), [mutual\_info\_classif](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)

#### Recursive Feature Elimination

Recursive Feature Elimination (RFE) as its name suggests recursively removes features, builds a model using the remaining attributes, and calculates model accuracy. RFE is able to work out the combination of attributes that contribute to the prediction of the target variable.

Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features.

```
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn import datasets
# Load Dataset
dataset = datasets.load_iris()
# Support Vectore Machine classifier
svm = LinearSVC()
# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(svm, 2)
rfe = rfe.fit(dataset.data, dataset.target)
# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)
```

### Feature Extraction (Bonus)

> Feature extraction is very different from Feature selection: the former consists of transforming arbitrary data, such as text or images, into numerical features usable for machine learning. The latter is a machine learning technique applied to these features.

We’ve decided to show you a standard technique from _sklearn._

#### Loading Features from Dicts

The class [DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer) transforms lists of feature-value mappings to vectors.

In particular, it turns lists of mappings (dict-like objects) of feature names to feature values into _Numpy_ arrays or _scipy.sparse_ matrices for use with scikit-learn estimators.

While not particularly fast to process, Python’s dict has the advantages of being convenient to use, being sparse (absent features need not be stored), and storing feature names in addition to values.

```
measurements = [
    {'city': 'Milano', 'temperature': 33.},
    {'city': 'Torino', 'temperature': 12.},
    {'city': 'Roma', 'temperature': 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())
```

DictVectorizer is also a useful representation transformation for training sequence classifiers in Natural Language Processing (NLP).

#### Feature Hashing

Named as one of the best hacks in Machine Learning, [Feature Hashing](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing) is a fast and space-efficient way of vectorizing features, i.e. turning arbitrary features into indices in a vector or matrix. For this topic, sklearn’s documentation is exhaustive, you can find it in the link above.

### Feature Construction

There’s no strict recipe for Feature Construction, I personally consider it like 99% creativity. We’re gonna take a look at some use cases in the next lectures though.

So far, you should take a look at the Feature Extraction part of [this marvelous notebook](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367) from _Beluga,_ one of the Competition Grandmasters on Kaggle.

In the last two articles, we’ve been introducing Feature Engineering as a subsequent step to Feature Processing. As you can see, we’re building a data processing pipeline, indeed, the next step would be finding a way to deal with _missing values._ Stay tuned for the next article and don’t forget to take a look at our [Github page](https://github.com/MLJCUnito/ProjectX2020), you’ll find the code related to this series of articles.