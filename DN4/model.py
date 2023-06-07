import numpy as np
import pandas as pd

import ProGED as pg

from DN4_2_podatky import * 
from nltk import Nonterminal
from nltk import grammar
from nltk.grammar import ProbabilisticProduction
from nltk.grammar import PCFG
import nltk


linija = "#"*100
print(f"\n {linija} \n")

np.random.seed(1234)




class eqAlgo():

    def __init__(self, left, right, data, grammar, sample_size=1) -> None:
        self.right = right
        self.left = left
        self.data = data
        self.grammar = grammar
        self.sample_size = sample_size
        self.inital_sample_size = sample_size
        self.rull_list = self.grammar.grammar.productions()
        self.star_simbol = self.rull_list[0].lhs()
        self.path = None
        self.best_path = None
        self.best_grammar = grammar
        self.grammar_old = grammar


    def model(self, iter, happy, rate):
        
        for i in range(iter+1):
            try:
                self.generat_eq()
            except:
                self.sample_size += 10
                self.update_prob(method='fulfill', rate=-0.1)
                self.generat_eq()
            if self.loss < happy:
                # napaka je dovolj majhna in lahko zaklucimo algo
                print(f"Happy: Loss {self.loss}, Eq: {self.eq}")
                return self.loss, self.eq
            elif i == 0:
                self.initial_loss = self.loss
                self.loss_old = self.loss
                self.best_loss = self.loss
                self.best_eq = self.eq
                self.best_path = self.path
                self.best_index = i
                continue
            else:
                # BEST
                if self.loss < self.best_loss:
                    self.best_loss = self.loss
                    self.best_grammar = self.grammar
                    self.best_path = self.path
                    self.best_eq = self.eq
                    self.best_index = i

                # ce je prejsna enacba boljssa vzemi prejsno
                if self.loss_old < self.loss:                    
                    self.path = self.path_old
                    self.grammar = self.grammar_old
                    self.sample_size = self.inital_sample_size

                # ce je trenutna enacba boljsa vzemi trenutno
                elif self.loss_old >= self.loss:
                    self.sample_size = self.inital_sample_size

                # ce se prevec oddalimo od zacetnega koraka -> se postavi v korak najboljsega modela
                if (self.loss - self.initial_loss) > (self.initial_loss - self.best_loss)/6 or (self.best_index+10) < i:
                #if (self.loss - self.initial_loss) > (self.initial_loss - self.best_loss)/5 or (self.best_index+20) < i:
                    self.grammar = self.best_grammar
                    self.path = self.best_path
                    self.loss = self.best_loss
                    self.sample_size = 20
    
                self.grammar_old = self.grammar
                self.loss_old = self.loss
                self.path_old = self.path
                self.update_prob(method='fulfill', rate=rate)

            if i%100 == 0:
                print(f'i: {i:03d}, Loss: {self.loss:.4f}, Eq: {self.eq}')
        
        #print(self.initial_loss)
        #print(self.best_grammar)    
        print("Iteracija se je zakljucila!\n")
        print(f"Eq: {self.best_eq}")
        print(f"Loss: {self.best_loss}")
        return self.best_loss, self.best_eq


    def generat_eq(self):
        """ Generera enacbe in izlusci pomebne informacije:
                pot generiranja enacbe, 
                enabo, 
                napako enacbe
        """

        ED = pg.EqDisco(data=self.data, 
                sample_size=self.sample_size,
                lhs_vars=self.left,
                rhs_vars=self.right,
                generator = self.grammar)
        #print(self.grammar)
        ED.generate_models()
        model = ED.fit_models()
        try:
            i = self.index_best_eq(model)
        except:
            self.sample_size = 10
            self.generat_eq()
        if i == None:
            self.grammar = self.best_grammar
            self.generat_eq()
            return
        self.path_old = self.path
        self.path = grammar.code_to_expression(ED.models[i].info['code'])[1]
        self.eq = ED.models[i]
        self.loss = model[i].get_error()

    
    def index_best_eq(self, ED_model):
        """ Sprejme
                ED_model ... ED model iz katerega izluscim idex najboljsega generiranega modela
            Vrne: index
        """
        err = np.infty
        i = None
        for j, model in enumerate(ED_model):
            if model.get_error() < err:
                i = j
                err = model.get_error()
        return i
    

    def update_prob(self, method="", rate=0.1):
        """Sprjeme
                method ... metoda ki je lahko 'soft' ali 'fullfil' za posodabljanje poti
                rate ... za koliko naj posodobi verjetnosti
            Vrne: None
        """

        simpl_path = self.simple_grammar(self.path)
        
        if len(simpl_path)==0:
            print("Prazna pot --> verjetnosti nemorejo biti posdobljene!")
            return
        
        simpl_gram_soft = self.simple_grammar(self.rull_list)
        simpl_gram_fulfill = self.simple_grammar(self.rull_list)

        for var, rull_prob_path in simpl_path.items():
            rull_prob = simpl_gram_soft[var]
            rull_prob1 = simpl_gram_fulfill[var]
            simpl_gram_soft[var] = self.softmax_update(rull_prob, rull_prob_path, rate)
            simpl_gram_fulfill[var] = self.fulfill_update(rull_prob1, rull_prob_path, var, rate)
        
        if method == 'soft':
            self.grammar = self.create_grammar(simpl_gram_soft)
            #print(self.grammar)
        elif method == 'fulfill':
            self.grammar = self.create_grammar(simpl_gram_fulfill)
            #print(self.grammar)
        else:
            print('Napacen update! Verjetnosti se niso posodobile!')


    def softmax_update(self, old_dist, new_var, rate):
        """ Sprejme 
                old_dist ... porazdelitev 'vozlisca'
                new_var ... poti katerim je potrebno povecati verjetnost
                rate ... za koliko spremenimo verjetnosti
            Vrne: nove porazdelitve s pomocjo softmax posodobitve
        """
        
        m = len(new_var)
        n = len(old_dist)
        sum_exp = 0
        for rull, prob in old_dist.items():
            if rull in new_var:
                new_p = np.exp(prob + rate/m)
                sum_exp += new_p
                old_dist[rull] = new_p
            else:
                new_p = np.exp(prob - rate/(n-m))
                sum_exp += new_p
                old_dist[rull] = new_p

        return {rull : p/sum_exp for rull, p in old_dist.items()}


    def fulfill_update(self, old_dist, new_var, var, rate):
        """ Sprejme 
                old_dist ... porazdelitev 'vozlisca'
                new_var ... poti katerim je potrebno povecati verjetnost
                var ... kateri spremenljivki spreminjamo porazdelitve
                rate ... za koliko spremenimo verjetnosti
            Vrne: novo porazdelitv spremenljivke var 
        """
        m = len(new_var)
        n = len(old_dist)
        sum_p = 0
        for rull, prob in old_dist.items():
            if rull in new_var:
                new_p = 0 if (prob + rate/m) < 0 else (prob + rate/m)
                # onemogoci da bi ostali samo rekurzivni celni
                if new_p == 0 and var not in rull:
                    new_p = 0.001
                sum_p += new_p
                old_dist[rull] = new_p
            else:
                new_p = 0 if (prob - rate/(n-m)) < 0 else (prob - rate/(n-m))
                # onemogoci da bi ostali samo rekurzivni celni
                if new_p == 0 and var not in rull:
                    new_p = 0.001
                sum_p += new_p
                old_dist[rull] = new_p

        return {rull : p/sum_p for rull, p in old_dist.items()}


    def simple_grammar(self, gram):
        """ Sprejme 
                gram ... gramatiko tipa grammar production
            Vrne: gramatiko transormirano v slovar {sprem_1 : {pravilo_1 : p_1, ..., pravilo_n : p_n}, sprem_2 :{}, ...}
        """
        simpl_gram = {}
        for el in gram:
            prob = el.prob()
            rull = el.rhs()
            var = el.lhs()
            #print(ProbabilisticProduction(var, rull, prob=prob))
            try:
                simpl_gram[var][rull] = prob
            except:
                simpl_gram[var] = {rull:prob}
            #print(simpl_gram)
        return simpl_gram


    def create_grammar(self, simp_gram):
        """Sprejem
                simp_gram ... gramatiko v obliki slovarja
            Vrne: gramatiko tipa ProGED.GeneratorGrammmar
        """
        gram = []
        for var, dict_rull_prob in simp_gram.items():
            for rull, prob in dict_rull_prob.items():
                gram.append(ProbabilisticProduction(var, list(rull), prob=prob))
        
        return pg.GeneratorGrammar(PCFG(self.star_simbol, gram))
        



grammar = "E -> E '+' F [0.3]| E '-' F [0.3]| F [0.4] \n"
grammar += "F -> F '*' T [0.2]| F '/' T [0.4]| T [0.4] \n"
grammar += "T -> V [0.4]| '('E')' [0.3]| 'sin' '('E')'[0.3] \n"
#grammar += "V -> 'x1' [0.2]| 'x2' [0.2]| 'x3' [0.2] | 'x4' [0.2] | 'x5' [0.2]"
grammar += "V -> 'x1' [0.5]| 'x2' [0.5]"

grammar = pg.GeneratorGrammar(grammar)
l_vars = ["y"]
r_vars = ["x1", "x2"]
#r_vars = ["x1", "x2", "x3", "x4", "x5"]

#podatki = generiraj_enacbo_1(1000)
#podatki = generiraj_enacbo_2(1000)
podatki = generiraj_enacbo_3(1000)
al = eqAlgo(left=l_vars, right=r_vars, data=podatki, grammar=grammar, sample_size=1)
loss, _ = al.model(iter=1000, happy=0.01, rate=0.3)


#al.model(iter=1000, happy=0.1, rate=0.4)

# Nasel enacbo 2 v cca 600 iteracij:
# al.model(iter=1000, happy=0.1, rate=0.4) 



