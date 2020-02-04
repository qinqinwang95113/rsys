from recsys_framework.data_manager.reader import *
from recsys_framework.data_manager.splitter import *
from recsys_framework.data_manager.dataset_postprocessing import *
import recsys_framework.utils.menu as menu
import random


def create_dataset(save_folder_path=None, random_seed=True):
    """
    Function able to: load, postprocess and split and save a dataset
    """
    if random_seed == True:
        random_seed = random.randint(1,500)
    else:
        random_seed = 42
    print(random_seed)
    dataset_name = menu.single_choice('Select the dataset you want to create',
                       ['Movielens100KReader', 'Movielens1MReader', 'LastFMHetrec2011Reader', 'BookCrossingReader', 'CiteULike_aReader',
                        'Movielens20MReader','NetflixPrizeReader', 'SpotifyChallenge2018Reader', 'YelpReader', 'EpinionsReader', 'PinterestReader'])

    dataset_reader = eval(dataset_name)()
    dataset = dataset_reader.load_data()

    implicitization = menu.yesno_choice('Perform Implicitization?')

    if implicitization == 'y':
        threshold = float(input('insert the threshold for Implicitization\n'))
        assert threshold <= dataset.get_URM().data.max(), 'selected threshold is too high!'
        implicitizer = ImplicitURM(int(threshold))
        dataset = implicitizer.apply(dataset)
        print('implicitization complete!')

    dataset_name = menu.single_choice('which kind of splitting?',
                       ['Holdout'])

    if dataset_name == 'Holdout':
        print('NOTE the percentage have to sum to 1\n')
        train_perc = float(input('insert train percentage\n'))
        val_perc = float(input('insert validation percentage\n'))
        test_perc = float(input('insert test percentage\n'))
        splitter = Holdout(train_perc=train_perc, validation_perc=val_perc, test_perc=test_perc)
    elif dataset_name == 'KFold':
        n_folds = int(input('how many folds?\n'))
        train_perc = float(input('which percentage of the dataset you want to divide in {} folds?\n'.format(n_folds)))
        splitter = WarmItemsKFold(n_folds=n_folds, percentage_initial_data_to_split=train_perc)

    save_dataset = menu.yesno_choice('Save the Dataset?\n')
    if save_dataset == 'y':
        splitter.save_split(splitter.split(dataset), filename_suffix='', save_folder_path=save_folder_path)


if __name__ == '__main__':
    create_dataset(random_seed=False)
