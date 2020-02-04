#from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
#from LatentFactorSimilarity.SIMCFRecommender import SIMCFRecommender

#from GraphBased.RP3beta_ML import RP3betaRecommender_ML

from recsys_framework.recommender.knn import ItemKNNCF

from recsys_framework.data_manager.reader import BrightkiteReader

if __name__ == '__main__':

    # dataset = Movielens100KReader()
    # dataset = Movielens10MReader()
    # dataset = Movielens20MReader()
    # dataset = Movielens1MReader()
    # dataset = Movielens100KReader()
    # dataset = EpinionsReader()
    # dataset = NetflixPrizeReader()

    #
    #
    # dataset_object = SpotifySkipPredictionReader()
    # dataset_object.load_data()
    #
    #
    # from data_manager.DataReaderPostprocessing_User_sample import DataReaderPostprocessing_User_sample
    # dataset_object = DataReaderPostprocessing_User_sample(dataset_object, user_quota=0.1)
    # dataset_object.load_data()
    #
    #
    # exit()
    #

    from recsys_framework.data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out

    dataset_object = BrightkiteReader()

    dataSplitter = DataSplitter_leave_k_out(dataset_object, k_value=15, validation_set=True)
    dataSplitter.load_data()

    URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

    # dataset = DataReaderPostprocessing_K_Cores(dataset, k_cores_value=25)
    # dataset = DataReaderPostprocessing_User_sample(dataset, user_quota=0.3)
    #
    # dataset = DataReaderPostprocessing_Implicit_URM(dataset)
    #
    # dataset.load_data()
    #
    # URM_train = dataset.get_URM_all()
    # URM_test = URM_train.copy()
    #
    #
    # from data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold, DataSplitter_ColdItems_k_fold
    #
    # dataSplitter = DataSplitter_Warm_k_fold(dataset)
    # #dataSplitter = DataSplitter_ColdItems_k_fold(dataset)
    #
    # dataSplitter.load_data()
    #
    # URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()
    #

    # available_ICM = dataSplitter.get_dataReader_object().get_loaded_ICM_names()
    # ICM_train = dataSplitter.get_dataReader_object().get_ICM_from_name(available_ICM[0])


    #recommender = ItemKNNCBFRecommender(ICM_train, URM_train)
    recommender = ItemKNNCF(URM_train)
    #recommender = PureSVDRecommender(URM_train)
    #recommender = UserKNNCFRecommender(URM_train)
    #recommender = SLIM_BPR_Cython(URM_train, train_with_sparse_weights=False, symmetric=True, URM_validation=URM_validation)
    #recommender = SLIM_KarypisLab(URM_train)
    #recommender = FW_SIMILARITY_RMSE_Cython(URM_train, ICM, S_SLIM, recompile_cython=True)
    #recommender = ItemKNNCustomSimilarityRecommender()
    #recommender = MF_BPR_Cython(URM_train, recompile_cython=False)

    #recommender = MatrixFactorization_AsySVD_Cython(URM_train, URM_validation = URM_validation)
    #recommender = MatrixFactorization_BPR_Cython(URM_train, URM_validation = URM_validation)
    #recommender = MatrixFactorization_FunkSVD_Cython(URM_train, URM_validation = URM_validation)
    #recommender = FW_SIMILARITY_RMSE_Python(URM_train, ICM, S_SLIM)
    #recommender = FW_RATING_RMSE_Python(URM_train, ICM)
    #recommender = FW_RATING_RMSE_Cython(URM_train, ICM, recompile_cython=True)
    #recommender = ElasticNet(URM_train)
    #recommender = FunkSVD(URM_train)
    #recommender = MultiThreadSLIM_RMSE(URM_train)

    #recommender = SIMCFRecommender(URM_train)
    #recommender = SLIM_Structure_BPR_Cython(URM_train, ICM = ICM_train, URM_validation=URM_validation, recompile_cython=False)

    #recommender = P3alphaRecommender(URM_train)
    #recommender = RP3betaRecommender(URM_train)
    #recommender = RP3betaRecommender_ML(URM_train)
    #recommender = Random(URM_train)
    #recommender = TopPop(URM_train)
    #recommender = GlobalEffects(URM_train)

    #recommender = SLIM_KarypisLab(URM_train)


    #logFile = open("timingrecommend_results.txt", "a")
    # optimalParam = dataSplitter.dataReader.get_hyperparameters_for_rec_class(recommender.__class__)
    #
    #
    # recommender.fit(topK = 500, loss="bpr", epochs=3, init_type="zero",
    #                 structure_mode="similarity", validation_every_n=1)

    recommender.fit()

    #recommender.fit(validation_every_n=2, stop_on_validation=True, lower_validatons_allowed=2)
    #recommender.fit(early_stopping=0, URM_validation = URM_validation)
    #
    # recommender.fit(validation_every_n=5, epochs=200, learning_rate=0.001, lambda_1=0.0, lambda_2=0.0,
    #                 batch_size=5, sgd_mode="adagrad", init_type="random", structure_mode="similarity", loss="bpr", topK=500)
    # results_run = recommender.evaluateRecommendations(URM_test, at=10, exclude_seen=False)
    # print(results_run)
    #
    #recommender.save_model("./result_experiments/", namePrefix = "Spotify2018Reader_ItemKNNCFRecommender_track_similarity")
    #
    # recommender = ItemKNNCBFRecommender(ICM, URM_train)
    # recommender.load_model("./result_experiments/")
    #
    # results_run = recommender.evaluateRecommendations(URM_test, at=20, exclude_seen=True)
    # print(results_run)
    #
    # results_run = recommender.evaluateRecommendations(URM_test)
    # print(results_run)
    #
    #
    #
    # from Base.evaluation.metrics import Diversity
    #
    # recommender_cbf = ItemKNNCBFRecommender(ICM_train, URM_train)
    # recommender_cbf.fit()
    # custom_diversity = Diversity(1-recommender_cbf.W_sparse.toarray())

    from recsys_framework.evaluation import EvaluatorHoldout

    # evaluator = SequentialEvaluator(URM_test, [5,20])#, diversity_object=custom_diversity)
    #
    # results_run, results_run_string = evaluator.evaluateRecommender(recommender)
    #
    # print("Sequential:\n" + results_run_string)


    evaluator = EvaluatorHoldout([5], exclude_seen=True)#, diversity_object=custom_diversity)

    metric_handler = evaluator.evaluateRecommender(recommender, URM_test=URM_test)

    print("Original:\n" + metric_handler.get_results_string())

    #
    #
    #
    # from ExperimentalAlgs.DiversityRerankingRecommender import DiversityRerankingRecommender
    #
    # diversityReranking = DiversityRerankingRecommender(recommender)
    #
    # for item_count_pow in list(np.arange(0.05, 1.0, 0.05)):
    #
    #     diversityReranking.fit(cutoff=5, item_count_pow = item_count_pow)
    #
    #     results_run, results_run_string = evaluator.evaluateRecommender(diversityReranking)
    #
    #     print("Reranked pow {}:\n".format(item_count_pow) + results_run_string)
