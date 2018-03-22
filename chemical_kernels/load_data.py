import glob, os, re
import csv
from random import shuffle

random.seed(1)

# retrieve data based on input name
def read_csv(path, feature_name, target_name):
    data = [[], []]
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for line in reader:
            data[0].append(line[feature_name])
            data[1].append(float(line[target_name]))
    
    return (data)


def shuffle_data(data):
    pairs = list(zip(data[0], data[1]))
    shuffle(pairs)
    feature, target = zip(*pairs)
    rand_data = [list(feature), list(target)]
    return (rand_data)



def preprocess(path, new_file):

    for filename in glob.iglob(os.path.join(path, '*.xyz*')):
        os.rename(filename, folder + filename[-4:] + '.xyz')
        
    k = 0
    text = []
    for filename in glob.iglob(os.path.join(path, '*.xyz')):   
        with open (filename) as input:
            i = 0
            for line in input:
                if i == 0:
                    start = line
                if i != 0 and re.match(start, line):
                    break
                text.append(line)
                i = i + 1

    with open(new_file,"w") as f:
        for i in range(len(text)):
            f.write(text[i])


# input_path = '../data/after_shuffle.csv'
# feature_name = 'smiles'
# target_name = 'activation energy'

# data = read_csv(input_path, feature_name, target_name)

