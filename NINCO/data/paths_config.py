import os
ninco_folder = '/data/alswo5131/maha-norm/NINCO/NINCO'
repo_path = '/data/alswo5131/maha-norm/NINCO'

dset_location_dict = {
    'NINCO': ninco_folder
}

root_folder_0 = '/data/alswo5131/ILSVRC/Data'
dset_location_suffix_dict_0 = {
    'ImageNet1K': 'CLS-LOC',
}

additional_dset_location_dict_0 = {k: os.path.join(root_folder_0, v) for k,v in dset_location_suffix_dict_0.items()}
dset_location_dict.update(additional_dset_location_dict_0)

dataset_csvs = {
    'NINCO_OOD_classes': 'data/NINCO_OOD_classes.csv',
    'NINCO_OOD_unit_tests': 'data/NINCO_OOD_unit_tests.csv',
    'NINCO_popular_datasets_subsamples': 'data/NINCO_popular_datasets_subsamples.csv',
}

naming_csvs = {
    'NINCO_class_names': 'data/NINCO_class_names.csv',
}