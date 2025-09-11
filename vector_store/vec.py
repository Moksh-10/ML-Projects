import numpy as np
from typing import *
from enum import Enum
from functools import wraps
import time
from sentence_transformers import SentenceTransformer

class sim_metric(Enum):
    EUC = 0
    COS = 1

def timeit(func):
    @wraps(func)
    def inner(*args, **kwargs):
        st = time.time()
        e = func(*args, **kwargs)
        en = time.time() - st
        return e, en * 1000
    return inner


class vec_store:
    def __init__(self, docs: List[str], embedder: SentenceTransformer = None, sim_metric: sim_metric = sim_metric.EUC) -> None:
        self.docs = np.array(docs)
        self.emb = None or embedder

        self._store: np.ndarray = None
        self.sim_metric = sim_metric
        self._set_sim_func()

    def set_metric(self, metric: sim_metric):
        assert isinstance(metric, sim_metric)
        self.sim_metric = metric
        self._sim_func = self._set_sim_func()

    def _set_sim_func(self):
        if self.sim_metric == sim_metric.EUC:
            self._sim_func = self._dist_euclidean__
        elif self.sim_metric == sim_metric.COS:
            self._sim_func = self._cosine__
        else:
            NotImplementedError(f"{self.sim_metric} not done")
