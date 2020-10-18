import gym
import numpy as np
from retro.examples.discretizer import Discretizer

class FZeroDiscretizer(Discretizer):
    def __init__(self, env):
        comandosSimples = [['B'],            #acelerar
                           ['LEFT', 'B'],    
                           ['RIGHT', 'B'],
                           ['X', 'Y'],       #frear
                           ['A', 'B']]      #turbo
        comandosSalto = [['LEFT'], ['RIGHT'], [''],    #o carro tem momento
                          ['UP'],                        #ficar mais tempo no ar
                          ['UP','LEFT'], ['UP', 'RIGHT'],
                          ['DOWN'],                     #ficar menos tempo no ar
                          ['DOWN', 'LEFT'], ['DOWN', 'RIGHT']]
        comandosDrift =  [['L'], ['R'],                  #Inclina o carro
                          ['L', 'LEFT'], ['R', 'RIGHT'],
                          ['L', 'B'], ['R', 'B'],
                          ['L', 'LEFT', 'B'], ['R', 'RIGHT','B']] 
        comandos = [comandosSimples[0] + comandosSalto[0] + comandosDrift[0]]
        super().__init__(env=env, combos=comandos)
        
        
class SMarioKartDiscretizer(Discretizer):
    def __init__(self, env):
        comandosSimples = [['B'],            #acelerar
                           ['LEFT', 'B'],    
                           ['RIGHT', 'B'],
                           ['Y'],       #frear
                           ['A']]        #usar item
        comandosDrift =  [['L', 'B'],                  #Pular
                          ['L', 'LEFT'], ['R', 'RIGHT'],  #drift
                          ['L', 'LEFT', 'B'], ['R', 'RIGHT','B']]
        comandos = [*comandosSimples + comandosDrift]
        super().__init__(env=env, combos=comandos)