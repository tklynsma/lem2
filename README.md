# LEM2 rule induction
## Description
A python2.7 implementation of LEM2 (Learning from Examples Module, version 2): a rule induction algorithm based on rough set theory.

## Usage
```python
from lem2_classifier import LEM2Classifier

X_train, y_train, X_test, y_test = ...
lem2 = LEM2Classifier()
lem2.fit(X_train, y_train)
predictions = lem2.predict(X_test)
```

## Example
LEM2 requires a data set containing only categorical attributes and decision values labeled by an expert. One such dataset is the Tic-Tac-Toe Endgame Dataset found [here](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame). First, load the data:

```python
>>> import numpy as np
>>> data = np.loadtxt("tic-tac-toe.data", dtype='str', delimiter=',')
>>> X, y = data[:,0:-1], data[:,len(data[0])-1]
```

Next, fit the data:

```python
>>> from lem2_classifier import LEM2Classifier
>>> lem2 = LEM2Classifier()
>>> lem2.fit(X, y)
>>> lem2.print_rules(class_name="win for x")
```