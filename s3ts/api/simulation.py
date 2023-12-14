#/usr/bin/env python3
# -*- coding: utf-8 -*-

from s3ts.api.ts2sts import inf_random_STS, inf_balanced_STS
from s3ts.api.encodings import compute_DM, compute_GM

from typing import Generator, Literal, Any
from collections import deque
import numpy as np

class StreamSimulator(Generator):

    """ Streaming time series simulator. """

    def __init__(self, 
                 X: np.ndarray, 
                 Y: np.ndarray, 
                 patts: np.ndarray,
                 wdw_len: int, 
                 wdw_str: int,
                 sts_method: Literal["random", "balanced"],
                 image_method: Literal["df", "gf", None] = None,
                 image_args: dict[str, Any] = {},
                 random_seed: int = 42,
                 discard: int = 100
                 ) -> None:

        # save parameters as attributes
        super().__init__()

        self.X, self.Y, self.patts = X, Y, patts
        self.wdw_len, self.wdw_str = wdw_len, wdw_str
        self.npatts, self.lpatts = patts.shape[0], patts.shape[2]
        self.image_method, self.image_args = image_method, image_args

        if sts_method == "random":
            self.STS_gen = inf_random_STS(X, Y, seed=random_seed)
        elif sts_method == "balanced":
            self.STS_gen = inf_balanced_STS(X, Y, seed=random_seed)

        maxlen = wdw_len*wdw_str
        self.STS, self.SCS = deque(maxlen=maxlen), deque(maxlen=maxlen)
        self.DM = deque(maxlen=maxlen)

        self.prev_col = None
        for _ in range(discard+maxlen):
            xi, yi = next(self.STS_gen) # type: ignore
            xi = np.expand_dims(xi, 1)
            self.STS.append(xi)
            self.SCS.append(yi)
            dmc = np.zeros((self.npatts, self.lpatts, 1))
            if self.image_method is not None:
                if self.image_method == "df":
                    dmc = compute_DM(STS=xi, patts=self.patts,
                        prev_col=self.prev_col, **self.image_args)
                if self.image_method == "gf":
                    dmc = compute_GM(STS=xi, patts=self.patts,
                        prev_col=self.prev_col, **self.image_args)
            self.DM.append(dmc)
            self.prev_col = np.squeeze(dmc, 2)

    def send(self, _) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xi, yi = next(self.STS_gen) # type: ignore
        xi = np.expand_dims(xi, 1)
        self.STS.append(xi)
        self.SCS.append(yi)

        dmc = np.zeros((self.npatts, self.lpatts, 1))
        if self.image_method is not None:
            if self.image_method == "df":
                dmc = compute_DM(STS=xi, patts=self.patts,
                    prev_col=self.prev_col, **self.image_args)
            if self.image_method == "gf":
                dmc = compute_GM(STS=xi, patts=self.patts,
                    prev_col=self.prev_col, **self.image_args)
        self.DM.append(dmc)
        self.prev_col = np.squeeze(dmc, 2)

        idx = self.wdw_len*self.wdw_str-1
        pxi = np.hstack(self.STS)[:,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]
        pyi = np.array(self.SCS)[0]
        pdm = np.squeeze(np.swapaxes(
            np.array(self.DM),0,3))[:,:,idx - self.wdw_len*self.wdw_str+1:idx+1:self.wdw_str]

        return pxi, pyi, pdm

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def samples_from_simulator(
        sim: StreamSimulator,          
        nsamps: int,
        mode: Literal["det", "prob"] = "det",
        every_n: int = 10,
        acc_prob: float = 0.1,
        seed: int = 42,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Creates a dataset of samples from a stream simulator. """
    series, labels, frames  = [], [], []
    si, li, fi = None, None, None
    if mode == "det":
        for _ in range(nsamps):
            for _ in range(every_n):
                si, li, fi = next(sim)
            series.append(si)
            labels.append(li)
            frames.append(fi)
    if mode == "prob":
        rng = np.random.default_rng(seed=seed)
        while len(labels) < nsamps:
            si, li, fi = next(sim)
            if rng.uniform() < acc_prob: 
                series.append(si)
                labels.append(li)
                frames.append(fi)
                nsamps += 1
    return np.array(series), np.array(labels), np.array(frames)
