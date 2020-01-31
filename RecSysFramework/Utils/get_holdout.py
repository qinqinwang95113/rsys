
from RecSysFramework.Utils import menu
from RecSysFramework.DataManager.Reader.Movielens100KReader import Movielens100KReader
from RecSysFramework.DataManager.Reader.Movielens1MReader import Movielens1MReader
from RecSysFramework.DataManager.Reader.LastFMHetrec2011Reader import LastFMHetrec2011Reader
from RecSysFramework.DataManager.Reader.BookCrossingReader import BookCrossingReader
from RecSysFramework.DataManager.DatasetPostprocessing.ImplicitURM import ImplicitURM
from RecSysFramework.DataManager.DatasetPostprocessing.KCore import KCore
from RecSysFramework.DataManager.DatasetPostprocessing.LongQueueAnalysis import LongQueueAnalysis
from RecSysFramework.DataManager.Splitter import Holdout
from RecSysFramework.DataManager.Reader.CiteULikeReader import CiteULike_aReader
from RecSysFramework.DataManager.Reader.Movielens20MReader import Movielens20MReader
from RecSysFramework.DataManager.Reader.EpinionsReader import EpinionsReader
from RecSysFramework.DataManager.Reader.PinterestReader import PinterestReader

def retrieve_train_validation_test_holdhout_dataset(dataset_name = None):
    if dataset_name == None:
        dataset_name = menu.single_choice('Select the dataset you want to create',
                        ['Movielens100KReader', 'Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader', 'CiteULike_aReader',
                         'Movielens20MReader', 'EpinionsReader', 'PinterestReader'])

    if dataset_name == 'Movielens100KReader':
        reader = Movielens100KReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(3), KCore(5, 5)])
    if dataset_name == 'Movielens1MReader':
        reader = Movielens1MReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(3), KCore(5, 5)])
    if dataset_name == 'Movielens20MReader':
        reader = Movielens20MReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(3), KCore(5, 5)])
    if dataset_name == 'BookCrossingReader':
        reader = BookCrossingReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(6), KCore(5, 5)])
    if dataset_name == 'CiteULike_aReader':
        reader = CiteULike_aReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[KCore(5, 5)])
    if dataset_name == 'LastFMHetrec2011Reader':
        reader = LastFMHetrec2011Reader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(1), KCore(5, 5)])
    if dataset_name == 'EpinionsReader':
        reader = EpinionsReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(3), KCore(4, 4)])
    if dataset_name == 'PinterestReader':
        reader = PinterestReader()
        h = Holdout(train_perc=0.6, validation_perc=0.2, test_perc=0.2)
        train, test, validation = h.load_split(
            reader, postprocessings=[ImplicitURM(1), KCore(5, 5)])


    return train, test, validation, dataset_name.replace('Reader', '').replace('CiteULike_a', 'CiteULike-a')
