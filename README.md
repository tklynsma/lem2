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
LEM2 requires a data set containing only categorical attributes and decision values labeled by an expert. One such dataset is the well-known playtennis dataset (included in examples). First, load the data and separate the attribute values from the decision values:

```python
>>> import numpy as np
>>> data = np.loadtxt("examples/playtennis.data", dtype='str', delimiter=',')
>>> X, y = data[:,0:-1], data[:,len(data[0])-1]
```

Next, fit the data using the LEM2 classifier:

```python
>>> from lem2_classifier import LEM2Classifier
>>> lem2 = LEM2Classifier()
>>> lem2.fit(X, y)
>>> lem2.print_rules(attr_names=["outlook","temperature","humidity","wind"], class_name="play tennis")
Rule: (play tennis, yes) <- (humidity, normal), (wind, weak) [Acc. 100.0, Cov. 28.6]
Rule: (play tennis, yes) <- (outlook, overcast) [Acc. 100.0, Cov. 28.6]
Rule: (play tennis, no) <- (outlook, sunny), (humidity, high) [Acc. 100.0, Cov. 21.4]
Rule: (play tennis, yes) <- (outlook, rain), (wind, weak) [Acc. 100.0, Cov. 21.4]
Rule: (play tennis, yes) <- (outlook, sunny), (humidity, normal) [Acc. 100.0, Cov. 14.3]
Rule: (play tennis, no) <- (outlook, rain), (wind, strong) [Acc. 100.0, Cov. 14.3]
```