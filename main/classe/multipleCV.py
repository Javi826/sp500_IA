#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 22:12:21 2024

@author: javi
"""
import numpy as np

import numpy as np
import pandas as pd

class MultipleTimeSeriesCV:
    
    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X['date'].unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_date = days[i * self.test_length]
            test_start_date = days[i * self.test_length + self.test_length - 1]
            train_end_date = days[i * self.test_length + self.test_length + self.lookahead - 1]
            train_start_date = days[i * self.test_length + self.test_length + self.lookahead +
                                   self.train_length - 1]
            split_idx.append([train_start_date, train_end_date, test_start_date, test_end_date])

        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = X[(X['date'] > train_start) & (X['date'] <= train_end)].index
            test_idx = X[(X['date'] > test_start) & (X['date'] <= test_end)].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits

