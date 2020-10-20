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
        comandosSalto = [['LEFT'], ['RIGHT'],    #o carro tem momento
                          ['UP'],                        #ficar mais tempo no ar
                          ['UP','LEFT'], ['UP', 'RIGHT'],
                          ['DOWN'],                     #ficar menos tempo no ar
                          ['DOWN', 'LEFT'], ['DOWN', 'RIGHT']]
        comandosDrift =  [['L'], ['R'],                  #Inclina o carro
                          ['L', 'LEFT'], ['R', 'RIGHT'],
                          ['L', 'B'], ['R', 'B'],
                          ['L', 'LEFT', 'B'], ['R', 'RIGHT','B']] 
        comandos = [*comandosSimples + comandosSalto + comandosDrift]
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

class MegaManDiscretizer(Discretizer):
    def __init__(self, env):
        comandosSimples = [['B'], ['RIGHT'], ['LEFT'], ['A']] 
        comandosSalto =  [['B', 'A'],['RIGHT','A'], ['LEFT','A'],
                         ['RIGHT','A', 'B'],['LEFT','A', 'B']]
        comandos = [*comandosSimples+comandosSalto]
        super().__init__(env=env, combos=comandos)