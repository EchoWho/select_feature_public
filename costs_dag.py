import numpy as np
from copy import deepcopy

class CostsManager(object):

    # cost_list   a list of cost of feat in addition to immediate dependencies;
    # dep_list    a map from a feat index to the list of immediate dependencies;
    # feat_map    maps usable final feat idx to dep_list feat_idx
    def __init__(self, cost_list, dep_list=None, feat_map=lambda x:x):
        if dep_list is None:
            dep_list = {}
        for g in range(cost_list.shape[0]):
            if not dep_list.has_key(g):
                dep_list[g] = []
        self.feat_map = feat_map
        self.dep_list = deepcopy(dep_list)
        self.has_computed = np.zeros(cost_list.shape[0], dtype=bool)
        self.cost_list = cost_list

    # feat  index of the feature we choose
    def choose(self, feat):
        feat = self.feat_map(feat)
        for g in self.dep_list[feat]:
            self.has_computed[g] = True
        self.has_computed[feat] = True

    def cost_of(self, feat):
        feat = self.feat_map(feat)
        if self.has_computed[feat]:
            return 0
        cost = self.cost_list[feat]
        for g in self.dep_list[feat]:
            if not self.has_computed[g]:
                cost += self.cost_list[g]
        return cost

    def total_cost(self):
        return np.sum(self.cost_list[np.where(self.has_computed)[0]])

    def total_possible_cost(self):
        return np.sum(self.cost_list)
