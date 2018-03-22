from load_data import *

input_path = '../data/after_shuffle.csv'
feature_name = 'smiles'
target_name = 'activation energy'

data = read_csv(input_path, feature_name, target_name)
