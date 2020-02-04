#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/2018

@author: Maurizio Ferrari Dacrema
"""

import unittest
import os, shutil


from recsys_framework.recommender.non_personalized import TopPop, Random, GlobalEffects
from recsys_framework.recommender.knn import UserKNNCF
from recsys_framework.recommender.knn import ItemKNNCF
from recsys_framework.recommender.SLIM.BPR import SLIM as SLIM_BPR
from recsys_framework.recommender.SLIM.ElasticNet import SLIM as SLIM_RMSE
from recsys_framework.recommender.GraphBased.P3alphaRecommender import P3alpha
from recsys_framework.recommender.GraphBased.RP3betaRecommender import RP3beta

from recsys_framework.recommender.MatrixFactorization import BPRMF, FunkSVD, AsySVD
from recsys_framework.recommender.MatrixFactorization import PureSVD
from recsys_framework.recommender.MatrixFactorization import IALS
from recsys_framework.recommender.MatrixFactorization import NMF


from recsys_framework.evaluation.evaluator import EvaluatorHoldout
from recsys_framework.data_manager.reader import Movielens1MReader
from recsys_framework.data_manager.splitter import Holdout
from recsys_framework.utils import EarlyStoppingModel


class RecommenderTestCase(unittest.TestCase):

    recommender_class = None

    def setUp(self):
        self.dataset = Movielens1MReader().load_data()
        self.splitter = Holdout(train_perc=0.8, test_perc=0.2, validation_perc=0.0)
        self.train, self.test = self.splitter.split(self.dataset)


    def common_test_recommender(self, recommender_class):

        temp_save_file_folder = self.dataset.get_complete_folder() + os.sep + "__temp__"
        os.makedirs(temp_save_file_folder, exist_ok=True)

        URM_train = self.train.get_URM()
        URM_test = self.test.get_URM()

        recommender_object = recommender_class(URM_train)

        if isinstance(recommender_object, EarlyStoppingModel):
            fit_params = {"epochs": 10}
        else:
            fit_params = {}

        recommender_object.fit(**fit_params)

        evaluator = EvaluatorHoldout([5], exclude_seen=True)
        metrics_handler = evaluator.evaluateRecommender(recommender_object, URM_test=URM_test)

        recommender_object.save_model(temp_save_file_folder, file_name="temp_model")

        recommender_object = recommender_class(URM_train)
        recommender_object.load_model(temp_save_file_folder, file_name="temp_model")

        evaluator = EvaluatorHoldout([5], exclude_seen=True)
        metrics_handler = evaluator.evaluateRecommender(recommender_object, URM_test=URM_test)

        shutil.rmtree(temp_save_file_folder, ignore_errors=True)


class RandomRecommenderTestCase(RecommenderTestCase):
    
    def test_recommender(self):
        self.common_test_recommender(Random)

class TopPopRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(TopPop)

class GlobalEffectsRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(GlobalEffects)

class UserKNNCFRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(UserKNNCF)

class ItemKNNCFRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(ItemKNNCF)

class P3alphaRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(P3alpha)

class RP3betaRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(RP3beta)

class SLIM_BPRRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(SLIM_BPR)

class SLIM_RMSERecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(SLIM_RMSE)

class BPRMFRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(BPRMF)

class FunkSVDRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(FunkSVD)

class AsySVDRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(AsySVD)

class PureSVDRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(PureSVD)

class NMFRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(NMF)

class IALSRecommenderTestCase(RecommenderTestCase):
    def test_recommender(self):
        self.common_test_recommender(IALS)


if __name__ == '__main__':
    unittest.main()
