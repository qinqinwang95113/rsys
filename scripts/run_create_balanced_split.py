#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/02/18

@author: Maurizio Ferrari Dacrema
"""

from data.DataSplitter import DataSplitter_ColdItems_WarmValidation
from data.NetflixEnhanced.NetflixEnhancedReader import NetflixEnhancedReader

from recsys_framework.recommender.knn import ItemKNNCBF

import numpy as np





max_delta_on_metric = 0.15
metric = "map"

split_accepted = False


while not split_accepted:


    dataReader_class = NetflixEnhancedReader


    dataSplitter = DataSplitter_ColdItems_WarmValidation(dataReader_class,
                                                         ICM_to_load=None,
                                                         force_new_split=True)

    URM_train = dataSplitter.get_URM_train()
    URM_validation = dataSplitter.get_URM_validation()
    URM_test = dataSplitter.get_URM_test()

    URM_validation.data[URM_validation.data<=3] = 0.0
    URM_validation.eliminate_zeros()

    URM_test.data[URM_test.data<=3] = 0.0
    URM_test.eliminate_zeros()

    test_items = dataSplitter.get_test_items()
    validation_items = dataSplitter.get_validation_items()
    train_items = dataSplitter.get_train_items()




    ICM_name = "ICM_editorial"
    ICM_dict = dataSplitter.get_split_for_specific_ICM(ICM_name)

    # ICM_train = ICM_dict[ICM_name + "_train"]
    # ICM_validation = ICM_dict[ICM_name + "_validation"]
    # ICM_test = ICM_dict[ICM_name + "_test"]

    ICM_train = ICM_dict[ICM_name + "_warm"]
    ICM_validation = ICM_dict[ICM_name + "_warm"]
    ICM_test = ICM_dict[ICM_name + "_global"]

    filter_for_train = test_items
    filter_for_validation = test_items
    filter_for_test = np.union1d(validation_items, train_items)



    dataReader = dataReader_class()
    optimalParam = dataReader.get_hyperparameters_for_rec_class(ItemKNNCBF)



    recommender = ItemKNNCBF(ICM_train, URM_train)
    recommender.fit(**optimalParam)

    result_train = recommender.evaluateRecommendations(URM_train, filterCustomItems=filter_for_train, exclude_seen=False, at=5, mode="sequential")
    print("ItemKNNCBFRecommender. Result URM_train: {}".format(result_train))

    # Use ICM_test as it contains all features for all items
    recommender = ItemKNNCBFRecommender(ICM_validation, URM_train)
    recommender.fit(**optimalParam)

    result_validation = recommender.evaluateRecommendations(URM_validation, filterCustomItems=filter_for_validation, exclude_seen=False, at=5, mode="sequential")
    print("ItemKNNCBFRecommender. Result URM_validation: {}".format(result_validation))


    recommender = ItemKNNCBFRecommender(ICM_test, URM_train)
    recommender.fit(**optimalParam)

    result_test = recommender.evaluateRecommendations(URM_test, filterCustomItems=filter_for_test, exclude_seen=False, at=5, mode="sequential")
    print("ItemKNNCBFRecommender. Result URM_test: {}".format(result_test))


    metric_values = [
        result_train[metric],
        result_validation[metric],
        result_test[metric]
    ]

    max_value = max(metric_values)
    min_value = min(metric_values)

    print("Metric '{}': max value is {:.4f}, min value is {:.4f}".format(metric, max_value, min_value))

    if max_value/min_value <= 1+max_delta_on_metric:
        split_accepted = True
        print("Split accepted!")

    else:
        print("Split not accepted")
