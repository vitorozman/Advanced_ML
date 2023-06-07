import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ProGED as pg
import nltk
from nltk import Nonterminal
from nltk import grammar
from nltk.grammar import ProbabilisticProduction
from nltk.grammar import PCFG

from DN4_2_podatky import * 
from model import *



def generate(grammar, podatki, happy, l_vars, r_vars, rate):
    AL = eqAlgo(left=l_vars, right=r_vars, data=podatki, grammar=grammar, sample_size=1)
    print('tki')
    df = {"iter": [], "Prob_algo":[], "Rand":[]}
    for i in range(100, 2001, 100):
        prob_loss, _ = AL.model(iter=i, happy=happy, rate=rate)

        ED = pg.EqDisco(data=podatki, 
                lhs_vars=-l_vars,
                rhs_vars=r_vars,
                sample_size=i)
        ED.generate_models()
        ED.fit_models()
        rand_loss = ED.get_results()[0].get_error()
        df["iter"].append(i)
        df["Prob_algo"].append(prob_loss)
        df["Rand"].append(rand_loss)
        break
    return df





grammar = "E -> E '+' F [0.3]| E '-' F [0.3]| F [0.4] \n"
grammar += "F -> F '*' T [0.2]| F '/' T [0.4]| T [0.4] \n"
grammar += "T -> V [0.4]| '('E')' [0.3]| 'sin' '('E')'[0.3] \n"


l_vars = ["y"]

# 1 ###################################################################################################
grammar1 = grammar + "V -> 'x1' [0.2]| 'x2' [0.2]| 'x3' [0.2] | 'x4' [0.2] | 'x5' [0.2]"
grammar1 = pg.GeneratorGrammar(grammar1)
print(grammar1)
podatki1 = generiraj_enacbo_1(1000)
r_vars1 = ["x1", "x2", "x3", "x4", "x5"]
df1 = generate(grammar1, podatki1, happy=0.5, l_vars=l_vars, r_vars=r_vars1, rate=0.3)
print(df1)

# 2 ###################################################################################################
grammar2 = grammar + "V -> 'x1' [0.5]| 'x2' [0.5]"
grammar2 = pg.GeneratorGrammar(grammar2)
podatki2 = generiraj_enacbo_2(1000)
r_vars2 = ["x1", "x2"]
df2 = generate(grammar2, podatki2, happy=0.1, l_vars=l_vars, r_vars=r_vars2, rate=0.4)

# 3 ###################################################################################################
grammar3 = grammar + "V -> 'x1' [0.5]| 'x2' [0.5]"
grammar3 = pg.GeneratorGrammar(grammar3)
podatki3 = generiraj_enacbo_3(1000)
r_vars3 = ["x1", "x2"]
df3 = generate(grammar3, podatki3, happy=0.01, l_vars=l_vars, r_vars=r_vars3, rate=0.3)







#podatki = generiraj_enacbo_2(1000)
#podatki = generiraj_enacbo_3(1000)