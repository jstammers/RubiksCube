'''
Wraps the Cube class into an OpenAI environment
'''
from gym import Env,Space
from Cube import RubiksCube
import numpy as np
import matplotlib.animation as animation

class CubeEnv(Env):
    '''
    Initialises a Cube environment
    '''
    def __init__(self,n=3):
        self.cube = RubiksCube(n)
        self.action_space = CubeActionSpace(self.cube)
        self.observation_space = self.cube._cube
        self.moves = self.action_space.moves
        self.metadata = {"render.modes":["human","rgb_array"]}
        self.score = self.cube.score_similarity()
        self.num_moves = 0
        reward_range = 0,1
    
    def _step(self,action):
        '''
        Implements a single action. This moves the cube using a given state
        '''
        index = np.argmax(action)
        self.cube.rotate_cube(*self.moves[index])
        observation = self.cube._cube
        score = self.cube.score_similarity()
        
        reward = np.max(score,axis=1)-np.max(self.score,axis=1)
        self.score = score
        done = np.mean(score) == 1
        self.num_moves+=1
        return observation,np.mean(reward),done,self.num_moves
        
    def _reset(self):
        return self.cube.reset(100)
        
    def _render(self,mode="rgb_array",close=False):
        if mode=="rgb_array":
            return self.cube.cube_colours()
        if close:
            return None
            
    def _close(self):
        return None
    
    def _seed(self,seed=None):
        if seed:
            np.random.RandomState(seed)
            
    def init_figure(self):
        self.cube.show_layout()
        
class CubeActionSpace(Space):
    
    def __init__(self,cube):
        self.cube = cube
        
        self.moves = []
        for i in range(self.cube.n):
            for j in range(self.cube.n):
                for k in [-1,1]:
                    self.moves.append((i,j,k))
        self.moves = np.array(self.moves)
        self.shape = self.moves.shape
        self.high = len(self.moves)/2
        self.low = -len(self.moves)/2
                    
    def sample(self):
        return np.random.randint(len(self.moves))
        
    def contains(self,x):
        return x in range(len(self.moves))
            
        
class CubeObservationSpace(Space):
    
    def __init__(self,cube):
        self.cube = cube
        self.obvs = [self.cube.score_similarity()]
        
    def sample(self):
        return 0
    def contains(self,x):
        return x == 0