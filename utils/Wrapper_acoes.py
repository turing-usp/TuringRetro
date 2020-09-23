
class OneHotDecoder_geral(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))
    
    def action(self, act):
        return self._decode_discrete_action[act].copy()


class FZeroDiscretizer(OneHotDecoder_geral):
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
        
        
class SMarioKartDiscretizer(OneHotDecoder_geral):
    def __init__(self, env):
        comandosSimples = [['B'],            #acelerar
                           ['LEFT', 'B'],    
                           ['RIGHT', 'B'],
                           ['Y'],       #frear
                           ['A']]        #usar item
        comandosDrift =  [['L' 'B']                  #Pular
                          ['L', 'LEFT'], ['R', 'RIGHT'],  #drift
                          ['L', 'LEFT', 'B'], ['R', 'RIGHT','B']] 
        comandos = [comandosSimples[0] + comandosDrift[0]]
        super().__init__(env=env, combos=comandos)