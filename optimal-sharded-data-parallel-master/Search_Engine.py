import numpy as np
from tqdm import trange

class SearchAlg():
    def __init__(self, max_mem=8200, layer_num=24, strategy_num=4) -> None:
        self.max_mem = max_mem + 1
        self.layer_num = layer_num
        self.strategy_num = strategy_num

        self._f = np.full((self.max_mem, strategy_num), 0, dtype=np.float64)
        
        self.v_data = None
        self.intra_cost = None

        self._mark = np.full((layer_num, self.max_mem, strategy_num), -1)

    
    def set_v_and_cost(self, v: np.ndarray, intra_layer_cost: np.ndarray):
        assert v.ndim == 2
        assert intra_layer_cost.ndim == 2

        assert v.shape[0] == self.layer_num
        assert v.shape[1] == self.strategy_num

        assert intra_layer_cost.shape[0] == self.layer_num
        assert intra_layer_cost.shape[1] == self.strategy_num

        self.v_data = v
        self.intra_cost = intra_layer_cost

    def fit(self):
        for i in range(self.layer_num):
            for v in range(self.max_mem - 1, -1, -1):
                for s in range(self.strategy_num):

                    if v < self.v_data[i, s]:
                        self._mark[i, v, s] = -1
                        self._f[v, s] = np.inf
                        continue

                    candidates = [self._f[v - self.v_data[i, s], si] for si in range(self.strategy_num)]
                    candidates = np.array(candidates) + self.intra_cost[i, s]

                    min_index = np.argmin(candidates)

                    self._mark[i, v, s] = min_index
                    self._f[v, s] = candidates[min_index]
        
        next_index, next_v = np.argmin(self._f[-1, :]), self.max_mem - 1
        total_cost = self._f[-1, next_index]

        if not total_cost < np.inf:
            return np.inf, None, -1

        res_list = [-1] * self.layer_num
        res_list[-1] = next_index

        for i in range(self.layer_num - 1, 0, -1):
            next_index, next_v = self._mark[i, next_v, next_index], next_v - self.v_data[i, next_index]
            res_list[i - 1] = next_index

        return total_cost, res_list, next_v - self.v_data[0, next_index]
