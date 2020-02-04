#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import traceback

from recsys_framework.recommender.knn import ItemKNNCBF
from recsys_framework.recommender.non_personalized import TopPop

from recsys_framework.evaluation import EvaluatorHoldout

from recsys_framework.data_manager.reader import Movielens100KReader
from recsys_framework.data_manager.reader import Movielens1MReader
from recsys_framework.data_manager.reader import Movielens10MReader
from recsys_framework.data_manager.reader import Movielens20MReader

from recsys_framework.data_manager.reader import EpinionsReader


def run_dataset(dataset_class):


    try:
        dataset_object = dataset_class()

        from recsys_framework.data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
        from recsys_framework.data_manager import DataReaderPostprocessing_K_Cores
        from recsys_framework.data_manager import DataReaderPostprocessing_Implicit_URM

        dataset_object = DataReaderPostprocessing_K_Cores(dataset_object, k_cores_value=5)
        dataset_object = DataReaderPostprocessing_Implicit_URM(dataset_object)
        #dataset_object.load_data()



        #dataSplitter = DataSplitter_Warm_k_fold(dataset_object)
        dataSplitter = DataSplitter_leave_k_out(dataset_object, k_value=5)

        dataSplitter.load_data()

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        #
        # dataSplitter = DataSplitter_ColdItems_k_fold(dataset_object)
        #
        # dataSplitter.load_data()
        #
        # URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
        #

        return

        evaluator = EvaluatorHoldout(URM_test, [5], exclude_seen=True)


        recommender = TopPop(URM_train)
        recommender.fit()
        _, results_run_string = evaluator.evaluateRecommender(recommender)

        log_file.write("On dataset {} - TopPop\n".format(dataset_class))
        log_file.write(results_run_string)
        log_file.flush()


        for ICM_name in dataSplitter.get_loaded_ICM_names():

            ICM_object = dataSplitter.get_ICM_from_name(ICM_name)

            recommender = ItemKNNCBFRecommender(ICM_object, URM_train)
            recommender.fit()
            _, results_run_string = evaluator.evaluateRecommender(recommender)

            log_file.write("On dataset {} - ICM {}\n".format(dataset_class, ICM_name))
            log_file.write(results_run_string)
            log_file.flush()


        log_file.write("On dataset {} PASS\n\n\n".format(dataset_class))
        log_file.flush()


    except Exception as e:

        print("On dataset {} Exception {}".format(dataset_class, str(e)))
        log_file.write("On dataset {} Exception {}\n\n\n".format(dataset_class, str(e)))
        log_file.flush()

        traceback.print_exc()


if __name__ == '__main__':

    # dataset_list = [
    #     # Movielens100KReader,
    #     # Movielens1MReader,
    #     Movielens10MReader,
    #     # Movielens20MReader,
    #     EpinionsReader,
    #     NetflixPrizeReader,
    #     # ThirtyMusicReader,
    #     YelpReader,
    #     SpotifyChallenge2018Reader,
    #     # AmazonElectronicsReader,
    #     # AmazonBooksReader,
    #     # AmazonAutomotiveReader,
    #     # NetflixEnhancedReader,
    #     XingChallenge2016Reader,
    #     XingChallenge2017Reader,
    #     BookCrossingReader,
    #     # TheMoviesDatasetReader,
    #     TVAudienceReader,
    #     LastFMHetrec2011Reader,
    #     DeliciousHetrec2011Reader,
    # ]


    log_file_name = "./run_test_datasets.txt"


    dataset_list = [
        Movielens100KReader,
        Movielens1MReader,
        Movielens10MReader,
        Movielens20MReader,
        EpinionsReader,
        # NetflixPrizeReader,
        # ThirtyMusicReader,
        # YelpReader,
        # SpotifyChallenge2018Reader,
        # AmazonElectronicsReader,
        # AmazonBooksReader,
        # AmazonAutomotiveReader,
        # NetflixEnhancedReader,
        # XingChallenge2016Reader,
        # XingChallenge2017Reader,
        # BookCrossingReader,
        # TheMoviesDatasetReader,
        # TVAudienceReader,
        # LastFMHetrec2011Reader,
        # DeliciousHetrec2011Reader,
    ]

    log_file = open(log_file_name, "a")

    for dataset_class in dataset_list:
        run_dataset(dataset_class)

