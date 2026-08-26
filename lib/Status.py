# !/usr/bin/env python3
from dataclasses import dataclass


@dataclass
class Status(object):
    NewBestSolution: bool
    ImproveCurrentSolution: bool
    AcceptedAsCurrentSolution: bool
    NIterationRecomputeWeights: int
    NIterationWithoutImprovement: int
    IterationId: int

    def __init__(self):
        self.NewBestSolution = False
        self.ImproveCurrentSolution = False
        self.AcceptedAsCurrentSolution = False
        self.NIterationRecomputeWeights = 0
        self.NIterationWithoutImprovement = 0
        self.IterationId = 0

