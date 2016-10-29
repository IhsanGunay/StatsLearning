'''
Created on Sep 16, 2015

@author: mbilgic
'''

import numpy as np
from glob import glob
from sklearn.feature_extraction.text import CountVectorizer
from time import time
from itertools import repeat
import re

def load_imdb(path, split_half=True, shuffle=True, random_state=42, vectorizer=CountVectorizer(min_df=5, max_df=1.0, binary=True)):
    
    print("Loading the imdb reviews data")
    
    train_neg_files = glob(path + r"/train/neg/*.txt")
    train_pos_files = glob(path + r"/train/pos/*.txt")
    
    train_corpus = []
    
    y_train = []
    
    for tnf in train_neg_files:
        with open(tnf, 'r', errors='replace') as f:
            line = f.read()
            train_corpus.append(line)
            y_train.append(0)
            
    for tpf in train_pos_files:
        with open(tpf, 'r', errors='replace') as f:
            line = f.read()
            train_corpus.append(line)
            y_train.append(1)
            
    test_neg_files = glob(path + r"/test/neg/*.txt")
    test_pos_files = glob(path + r"/test/pos/*.txt")
    
    test_corpus = []
    
    y_test = []
    
    for tnf in test_neg_files:
        with open(tnf, 'r', errors='replace') as f:
            test_corpus.append(f.read())
            y_test.append(0)
            
    for tpf in test_pos_files:
        with open(tpf, 'r', errors='replace') as f:
            test_corpus.append(f.read())
            y_test.append(1)
                
    print("Data loaded.")
    
    print("Extracting features from the training dataset using a sparse vectorizer")
    print("Feature extraction technique is {}.".format(vectorizer))
    t0 = time()

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_train))        
        y_train = y_train[indices]
        train_corpus_shuffled = [train_corpus[i] for i in indices]
        
        np.random.seed(random_state)
        indices = np.random.permutation(len(y_test))        
        y_test = y_test[indices]
        test_corpus_shuffled = [test_corpus[i] for i in indices]
        
    if split_half:
        split = int(len(y_train)/2) 

        y_val = np.copy(y_train[split:])
        y_train = np.copy(y_train[:split])
    
        val_corpus_shuffled = train_corpus_shuffled[split:]
        train_corpus_shuffled = train_corpus_shuffled[:split]

        X_train = vectorizer.fit_transform(train_corpus_shuffled)
        X_val = vectorizer.transform(val_corpus_shuffled)
        X_test = vectorizer.transform(test_corpus_shuffled)
        
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        X_val = X_val.tocsr()
    
    duration = time() - t0
    print("done in {}s".format(duration))
    print("n_samples: {}, n_features: {}".format(*X_train.shape), '\n')
    
    return X_train, y_train, X_val, y_val, X_test, y_test, train_corpus_shuffled, val_corpus_shuffled, test_corpus_shuffled

def ce_squared(T, probs):
    return ((T*probs)**2).sum()/len(probs)

class ColoredDoc(object):

    def __init__(self, doc, feature_names, coefs):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ")        
        for token in tokens:
            vocab_tokens = self.token_pattern.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    if self.coefs[vocab_index] > 0:
                        html_rep = html_rep + "<font color=blue> " + token + " </font>"
                    elif self.coefs[vocab_index] < 0:
                        html_rep = html_rep + "<font color=red> " + token + " </font>"
                    else:
                        html_rep = html_rep + "<font color=gray> " + token + " </font>"
                except:
                    html_rep = html_rep + "<font color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font color=gray> " + token + " </font>"
        return html_rep

class ColoredWeightedDoc(object):

    def __init__(self, doc, feature_names, coefs, binary = False):
        self.doc = doc
        self.feature_names = feature_names
        self.coefs = coefs
        self.binary = binary
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.abs_ranges = np.linspace(0, max([abs(coefs.min()), abs(coefs.max())]), 8)

    def _repr_html_(self):
        html_rep = ""
        tokens = self.doc.split(" ") 
        if self.binary:
            seen_tokens = set()       
        for token in tokens:
            vocab_tokens = self.token_pattern.findall(token.lower())
            if len(vocab_tokens) > 0:
                vocab_token = vocab_tokens[0]
                try:
                    vocab_index = self.feature_names.index(vocab_token)
                    
                    if not self.binary or vocab_index not in seen_tokens:
                        
                        if self.coefs[vocab_index] > 0: # positive word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] < self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=blue> " + token + " </font>"
                        
                        elif self.coefs[vocab_index] < 0: # negative word
                            for i in range(1, 7):
                                if self.coefs[vocab_index] > -self.abs_ranges[i]:
                                    break
                            html_rep = html_rep + "<font size = " + str(i) + ", color=red> " + token + " </font>"
                        
                        else: # neutral word
                            html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                        
                        if self.binary:    
                            seen_tokens.add(vocab_index)
                    
                    else: # if binary and this is a token we have seen before
                        html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
                except: # this token does not exist in the vocabulary
                    html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
            else:
                html_rep = html_rep + "<font size = 1, color=gray> " + token + " </font>"
        return html_rep
    
class TopInstances():

    def __init__(self, neg_evis, pos_evis, intercept=0):
        self.neg_evis = neg_evis
        self.pos_evis = pos_evis
        self.intercept = intercept
        self.total_evis = self.neg_evis + self.pos_evis
        self.total_evis += self.intercept
        self.total_abs_evis = abs(self.neg_evis) + abs(self.pos_evis)
        self.total_abs_evis += abs(self.intercept)
        
    def most_negatives(self, k=1):
        evi_sorted = np.argsort(self.total_evis)
        return evi_sorted[:k]
    
    def most_positives(self, k=1):
        evi_sorted = np.argsort(self.total_evis)
        return evi_sorted[-k:][::-1]
    
    def least_opinionateds(self, k=1):
        abs_evi_sorted = np.argsort(self.total_abs_evis)
        return abs_evi_sorted[:k]
    
    def most_opinionateds(self, k=1):
        abs_evi_sorted = np.argsort(self.total_abs_evis)
        return abs_evi_sorted[-k:][::-1]
    
    def most_uncertains(self, k=1):
        abs_total_evis = abs(self.total_evis)
        abs_total_evi_sorted = np.argsort(abs_total_evis)
        return abs_total_evi_sorted[:k]
    
    def most_conflicteds(self, k=1):
        conflicts = np.min([abs(self.neg_evis), abs(self.pos_evis)], axis=0)
        conflict_sorted = np.argsort(conflicts)
        return conflict_sorted[-k:][::-1]
    
    def least_conflicteds(self, k=1):
        conflicts = np.min([abs(self.neg_evis), abs(self.pos_evis)], axis=0)
        conflict_sorted = np.argsort(conflicts)
        return conflict_sorted[:k]

class ClassifierArchive():

    def __init__(self, ctrl_clf, best_clf, train_indices, modified_labels, vect):
        self.vect = vect
        self.type = type(best_clf)
        self.ctrl_clf = ctrl_clf
        self.classifiers = [best_clf]
        self.train_indices = [train_indices]
        self.modified_labels = [modified_labels]
        self.round_tags = [1]
        assert type(best_clf) == type(ctrl_clf)

    def __len__(self):
        return len(self.classifiers)

    def stats(self):
        print(self.type, "\n")
        print(self.round_tags, "\n")
        print(self.vect)

    def add_classifier(self, clf, train_indices, modified_labels, round_tag):
        self.classifiers.append(clf)
        self.train_indices.append(train_indices)
        self.modified_labels.append(modified_labels)
        self.round_tags.append(round_tag)
        assert self.type == type(clf)
        assert len(self.classifiers) == len(self.train_indices)
        assert len(self.classifiers) == len(self.round_tags)
        assert len(self.classifiers) == len(self.modified_labels)

    def rm_classifier(round_tag):
        i = round_tags.index(round_tag)
        classifiers.pop(i)
        train_indices.pop(i)
        modified_labels.pop(i)
        round_tags.pop(i)
        assert len(self.classifiers) == len(self.train_indices)
        assert len(self.classifiers) == len(self.round_tags)
        assert len(self.classifiers) == len(self.modified_labels)

def produce_modifications(X_train, y_train, train_indices, target_indices, X_val, y_val_na):
    for i in target_indices:

        if i in train_indices:
            mod0 = np.copy(y_train)
            mod0[i] = 1 - mod0[i]
            yield X_train, mod0, train_indices, X_val, y_val_na

            mod1 = list(train_indices)
            mod1.remove(i)
            yield X_train, y_train, mod1, X_val, y_val_na

        else:
            mod0 = list(train_indices)
            mod0.append(i)
            yield X_train, y_train, mod0, X_val, y_val_na

            mod1 = np.copy(y_train)
            mod1[i] = 1 - mod1[i]
            yield X_train, mod1, mod0, X_val, y_val_na

def test_modification(test, clf):
    X_train, y_train, train_indices, X_val, y_val_na = test
    
    clf.fit(X_train[train_indices],y_train[train_indices])
    new_error = ce_squared(y_val_na, clf.predict_proba(X_val))
    
    return new_error, y_train, train_indices

def modify_dataset(estimator, X_train, y_train, X_val, y_val, batch_size):
    y_val_na = y_val[:, np.newaxis]
    y_val_na = np.append(y_val_na, 1-y_val_na, axis=1)
    
    start_ind = 0
    end_ind = start_ind + batch_size

    clf = clone(estimator)
    clf.fit(X_train, y_train)
    
    best_error = ce_squared(y_val_na, clf.predict_proba(X_val))
    best_y_train = np.copy(y_train)
    best_train_indices = list(range(X_train.shape[0]))
   
    with ProcessPoolExecutor() as executor:
        while end_ind <= X_train.shape[0]:
            target_indices = range(start_ind, end_ind)
            mods = produce_modifications(X_train, best_y_train, best_train_indices, target_indices, X_val, y_val_na)
            
            test_results = list(executor.map(test_modification, mods, repeat(clf)))
            test_results.append((best_error, best_y_train, best_train_indices))
            best_error, best_y_train, best_train_indices = min(test_results, key=lambda x: x[0])
            
            print('Processed: {:5d} samples,\tcurrent error is {:0.4f}'.format(end_ind, best_error))
            start_ind += batch_size
            end_ind += batch_size
            
    return X_train[best_train_indices], best_y_train[best_train_indices]