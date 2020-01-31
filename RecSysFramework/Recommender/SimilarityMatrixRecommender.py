#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017

@author: Cesare Bernardis, Maurizio Ferrari Dacrema
"""

import numpy as np

from RecSysFramework.Recommender.DataIO import DataIO
from RecSysFramework.Recommender import Recommender
from tkinter.filedialog import askopenfilename


class SimilarityMatrixRecommender(Recommender):
    """
    This class refers to a BaseRecommender KNN which uses a similarity matrix, it provides two function to compute item's score
    bot for user-based and Item-based models as well as a function to save the W_matrix
    """


    def __init__(self, URM_train, *pos_args):
        super(SimilarityMatrixRecommender, self).__init__(URM_train)

        self._URM_train_format_checked = False
        self._W_sparse_format_checked = False
        self.load_model_path = 'EvaluationResults/'

    def load_model(self, folder_path, file_name=None, gui=False):
        """
        override the method to use a gui for select the filename
        :return:
        """
        if gui:
            file_name = askopenfilename()  # show an "Open" dialog box and return the path to the selected file
            file_name = file_name.split('EvaluationResults/')[1]
            file_name = file_name.split('.zip')[0]
            folder_path = self.load_model_path
        super(SimilarityMatrixRecommender, self).load_model(folder_path=folder_path, file_name=file_name)

    def _check_format(self):

        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._W_sparse_format_checked:

            if self.W_sparse.getformat() != "csr":
                self._print("PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format("W_sparse", "csr"))

            self._W_sparse_format_checked = True


    def _get_dict_to_save(self):
        return {"W_sparse": self.W_sparse}


class ItemSimilarityMatrixRecommender(SimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse).toarray()


        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


class UserSimilarityMatrixRecommender(SimilarityMatrixRecommender):

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_weights_array = self.W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_weights_array.dot(self.URM_train).toarray()

        item_scores = self._compute_item_score_postprocess_for_cold_users(user_id_array, item_scores)
        item_scores = self._compute_item_score_postprocess_for_cold_items(item_scores)

        return item_scores


