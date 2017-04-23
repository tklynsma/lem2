from collections import Counter

class LEM2Classifier:
    
    def fit(self, data, y):
        """
        Learn the decision rules from the given training set (data, y) using
        the LEM2 rule induction algorithm.

        Args:
            data ([[str]]): the list of instances/objects in the dataset.
            y ([str]): the list of decision values for each instance.
        """
        self._U = [tuple(x) for x in data] # The universe of objects/instances.
        self._A = range(len(data[0]))      # The list of attribute indices.
        self._D = list(set(y))             # The list of decision values.
        self._rules = []                   # The list of rules.
        
        for d in self._D:
            X = [u for i, u in enumerate(self._U) if y[i] == d]
            Y = [u for i, u in enumerate(self._U) if y[i] != d]
            lower_approximation = set(X) - set(Y)
            upper_approximation = set(X)
            
            # Find the certain rules:
            covering = self._LEM2(lower_approximation)
            certain = [(1, self._coverage(C), C, d) for C in covering]
            self._rules.extend(certain)
            
            # Find the possible rules:
            if lower_approximation != upper_approximation:
                covering = self._LEM2(upper_approximation)
                possible = [(self._accuracy(C, X), self._coverage(C), C, d) for C in covering]
                self._rules.extend(set(possible) - set(certain))
        
        self._rules = sorted(self._rules, reverse=True)
        self._majority_class = Counter(y).most_common()[0][0]
        
    def print_rules(self, attr_names=None, class_name='d', min_acc=0, min_cov=0):
        """
        Print the decision rules to console.

        Args:
            attr_names ([str]): the list of attribute names.
            class_name (str): the name of the decision value.
            min_acc (int): the minimum accuracy of printed rules (default 0).
            min_cov (int): the minimum coverage of printed rules (default 0).
        """
        for (acc, cov, conditions, decision) in self._rules:
            if acc * 100 >= min_acc and cov * 100 >= min_cov:
                print "Rule: ({}, {}) <-".format(class_name, decision),
                for i, (a, v) in enumerate(conditions):
                    comma = "" if i == 0 else "\b, "
                    if attr_names is None:
                        print (comma + "({}, {})").format(a, v),
                    else:
                        print (comma + "({}, {})").format(attr_names[a], v),
                print "[Acc. {0:.1f}, Cov. {1:.1f}]".format(100 * acc, 100 * cov)
    
    def predict(self, X, method='lers'):
        """
        Return a list of predictions for the instances in X.

        Args:
            X ([str] or [[str]]): one instance or a list of instances/objects to be predicted.
            method (str): the classification method (lers or first_fit).
        Returns:
            str or [str]: the prediction or list of predictions.
        """
        if not type(X[0]) is list:
            if method == 'first_fit':
                return self._first_fit(X)
            elif method == 'lers':
                return self._lers_classification(X)

        else:
            if method == 'first_fit':
                return [self._first_fit(x) for x in X]
            elif method == 'lers':
                return [self._lers_classification(x) for x in X]
            
        print "Invalid classification method."
        return []
        
    def _first_fit(self, x):
        """
        Return the decision value of the first rule for which x satisfies the
        conditions. If there is no such rule return the majority class.
        """
        for (_, _, conditions, decision) in self._rules:
            if all([x[a] == v for (a, v) in conditions]):
                return decision
        return self._majority_class
        
    def _lers_classification(self, x):
        """
        Classify the value of x using the LERS classification method.
        """
        c_matching = dict([(d, []) for d in self._D])
        p_matching = dict([(d, []) for d in self._D])
        complete_matching, partial_matching = False, False
        
        # Find the complete and partial matching rules for each decision value.
        for r in self._rules:
            _, _, conditions, decision = r
            match = [x[a] == v for (a, v) in conditions]
            if all(match):
                c_matching[decision].append(r)
                complete_matching = True
            elif not complete_matching and any(match):
                p_matching[decision].append(r)
                partial_matching = True
        
        # If there exists at least one complete matching rule return the decision
        # value d for which its matching rule set has the highest support.
        if complete_matching:
            return max(c_matching, key=lambda k: self._support(c_matching[k]))
            
        # If there exists at least one partially matching rule return the decision
        # value d for which its matching rule set has the highest support.
        elif partial_matching:
            return max(p_matching, key=lambda k: self._support(p_matching[k], case=x))
            
        # If all else fails return the majority class.
        return self._majority_class
        
    def _support(self, rules, case=None):
        """
        Compute the support of a set of rules as defined for the LERS classifica-
        tion system. The support of a ruleset is the sum of products of strength,
        specificity and matching factor. Return -1 if the ruleset is empty.
        """
        if not rules:
            return -1
        if case is None:
            return sum([acc * cov * len(C) for acc, cov, C, _ in rules]) 
        else:
            return sum([sum([case[a] == v for (a, v) in C]) * acc * cov 
                for acc, cov, C, _ in rules])
         
    def _accuracy(self, conditions, X):
        """
        Compute the accuracy of a rule: the fraction of instances (relative to
        those satisfying the conditions) for which the rule has been found to be 
        true. X is the list of instances with the correct decision value.
        """
        return len(self._block(conditions, base=X)) / float(len(self._block(conditions)))
    
    def _coverage(self, conditions):
        """
        Computes the coverage of a rule: the fraction of instances in the whole
        dataset that satisfy the conditions.
        """
        return len(self._block(conditions)) / float(len(self._U))
        
    def _block_s(self, a, v, base=None):
        """
        Computes a block [t] for t = (a, v) and base X, where [t] is the set
        of all cases from X such that attribute a has value v. If the base is
        unspecified the base will be the set of all instances U.
        """
        if base is None:
            base = self._U
        return [x for x in base if x[a] == v]
        
    def _block(self, T, base=None):
        """
        Computes a block [T] of a set of attribute-value pairs T, where [T] is 
        the intersection of the blocks [t] for all t = (a, v) in T. If T is 
        empty then [T] contains all the instances in the base (default U).
        """
        if base is None:
            base = self._U
        return reduce(lambda S, t : self._block_s(*t, base=S), T, base)
 
    def _best_pair(self, G, T):
        """
        Estimate the best attribute-value pair in T for representing the concept
        G. Select the pair t in T such that [t] with base G is maximum. If a tie
        occurs, select the pair t in T such that [t] with base U is maximum. If
        another tie occurs select the first such pair.
        """
        minimum, maximum = len(self._U), 0
        best = list(T)[0]
        
        for t in T:
            support = len(self._block_s(*t, base=G))
            
            if support > maximum:
                best, maximum = t, support
                
            elif support == maximum:
                total = len(self._block_s(*t))
                if total < minimum:
                    best, minimum = t, total

        return best
        
    def _LEM2(self, concept):
        """
        Given a lower or upper approximation of a concept, compute a local
        covering of the concept. If X is a lower approximation the derived rules
        from the local covering are certain, if X is an upper approximation the 
        derived rules from the local covering are possible.
        """
        targets  = concept.copy() # The set of instances not yet covered by any rules.
        covering = set()          # The set of rule-conditions.
        
        while targets:
            pairs      = set([(a, x[a]) for a in self._A for x in targets])
                                       # The set of all relevant attribute-value pairs.
            conditions = set()         # The set of conditions (attribute value pairs).
            block      = set(self._U)  # The set of all instances covered by the conditions.
            
            # While conditions is empty or inconsistent with the data:
            while not conditions or not block.issubset(concept):
                best    = self._best_pair(targets, pairs)
                conditions.add(best)
                block   = set(self._block_s(*best, base=block))
                targets = set(self._block_s(*best, base=targets))
                pairs   = set([(a, x[a]) for a in self._A for x in targets])
                pairs   = pairs - conditions
            
            # Prune the conditions (reduce to a minimal complex):
            for t in conditions.copy():
                block = set(self._block(conditions - set([t])))
                if block.issubset(concept):
                    conditions.remove(t)
        
            covering.add(tuple(sorted(conditions)))
            targets = concept - set.union(*[set(self._block(T)) for T in covering])
            
        # Prune the rules (reduce to a local covering):
        for T in covering.copy():
            covered = [set(self._block(S)) for S in covering - set([T])]
            if not concept and set.union(*covered) == concept:
                covering.remove(T)
                
        return covering
