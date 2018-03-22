from chemical_kernels.load_data import *
from chemical_kernels.gen_kernel import *
from chemical_kernels.neural_graph_fps import *


input_path = 'data/original.csv'
feature_name = 'smiles'
target_name = 'activation energy'

#load data
data = read_csv(input_path, feature_name, target_name)
y_true_b = data[1]
data = shuffle_data(data)
smiles, y_true = data[0], data[1]


neural_graph_fps(target_name, input_path, len(smiles))

# construct kernels and build models
subsequence = Kernel.from_smi(smiles, y_true, 'subsequence')
mismatch = Kernel.from_smi(smiles, y_true, 'mismatch')
linear_fps = Kernel.from_smi(smiles, y_true, 'linear_fps')
morgan_fps = Kernel.from_smi(smiles, y_true, 'morgan_fps')
random_walk = Kernel.from_file('data/cl_random_walk0.3.txt', y_true_b)

# compute accuracies (R2)
subseq_test_r2, subseq_r2 = subsequence.accuracy()
mismatch_test_r2, mismatch_r2 = mismatch.accuracy()
linear_test_r2, linear_r2 = linear_fps.accuracy()
morgan_test_r2, morgan_r2 = morgan_fps.accuracy()
random_walk_test_r2, random_walk_r2 = random_walk.accuracy()

# print results
print ("Test R2 for subsequence kernel is: {}.".format(subseq_test_r2))
print ("Test R2 for mismatch kernel is: {}.".format(mismatch_test_r2))
print ("Test R2 for daylight fingerprints is: {}.".format(linear_test_r2))
print ("Test R2 for morgan fingerprints is: {}.".format(morgan_test_r2))
print ("Test R2 for random walk kernel is: {}.".format(random_walk_test_r2))