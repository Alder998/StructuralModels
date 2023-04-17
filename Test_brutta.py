# C'è una formula analitica per farlo, quindi niente panico

import pandas as pd
import numpy as np
import MertonModel as mm
import matplotlib.pyplot as plt
import processes as pr
import JumpDiffusionModelClosedForm as jp
import montecarloJumpDiffisionDaO as mdao
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import MarketDataScraper as mkt
import modelComparison as model
import random as rn

#model.allModelsComparisonOnRealStock('IBE.MC', 1, freq='quarterly')


