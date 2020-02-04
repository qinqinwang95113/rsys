from recsys_framework.recommender.non_personalized import TopPop, Random

from recsys_framework.data_manager.reader import Movielens1MReader
from recsys_framework.data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from recsys_framework.evaluation import KFoldResultRepository


if __name__ == '__main__':


    dataSplitter = DataSplitter_Warm_k_fold(Movielens1MReader, allow_cold_users=True, ICM_to_load=None,
                                            apply_k_cores=1, force_new_split=False, split_folder_path="provakfold/")
    #dataSplitter = DataSplitter_ColdItems_k_fold(XingChallenge2017Reader, ICM_to_load=None, apply_k_cores=1, force_new_split=False, forbid_new_split=False)

    ICM_name = "ICM_all"

    result_repo_alg1 = KFoldResultRepository(len(dataSplitter))
    result_repo_alg2 = KFoldResultRepository(len(dataSplitter))


    from recsys_framework.evaluation import SequentialEvaluator

    for fold_index, (URM_train, URM_test) in dataSplitter:

        print("Processing fold {}".format(fold_index))

        evaluator = SequentialEvaluator(URM_test, [10], exclude_seen=True)
        #ICM_train = dataSplitter.get_ICM(ICM_to_load = ICM_name)

        recommender = TopPop(URM_train)
        recommender.fit()
        metric_handler = evaluator.evaluateRecommender(recommender)

        print(metric_handler.get_results_string())
        result_repo_alg1.set_results_in_fold(fold_index, metric_handler.get_results_dictionary()[10])


        recommender = Random(URM_train)
        recommender.fit()
        metric_handler = evaluator.evaluateRecommender(recommender)

        print(metric_handler.get_results_string())
        result_repo_alg2.set_results_in_fold(fold_index, metric_handler.get_results_dictionary()[10])


    result_repo_alg1.run_significance_test(result_repo_alg2)