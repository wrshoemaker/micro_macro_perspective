import config
import os, sys
import pickle
from itertools import combinations

import ete4


dataset_all= ['caporaso_et_al', 'david_et_al', 'poyet_et_al']

path = '%sphylo_dist_dict.pickle' % config.data_directory


def make_phylo_dist_dict():

    phylo_dist_dict = {}

    for dataset in dataset_all:

        sys.stderr.write("Running %s.....\n" % dataset)

        tree_path = "%s%s-seqtab-nochim-gut_muscle.tre" % (config.data_directory, dataset)
        #tree_path = "%spoyet_et_al-seqtab-nochim-gut_muscle.tre" % config.data_directory
        # quoted_node_names=False, 
        tree = ete4.Tree(tree_path)
        tips = [str(leaf.name) for leaf in tree.leaves()]
        tips_pairs = [tuple(sorted(pair)) for pair in combinations(tips, 2)]

        phylo_dist_dict[dataset] = {}
        for tips_pair in tips_pairs:
            phylo_dist_dict[dataset][tips_pair] = tree.get_distance(str(tips_pair[0]), str(tips_pair[1]))


    with open(path, 'wb') as outfile:
        pickle.dump(phylo_dist_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":

    make_phylo_dist_dict()

    