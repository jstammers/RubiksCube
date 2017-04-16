'''
Wraps the Cube class into an OpenAI environment
'''
from gym import Env,Space
from Cube import RubiksCube

class CubeEnv(Env):
    def __init__(self):
        self.cube = RubiksCube()
        self.action_space = CubeActionSpace(cube)
        self.observation_space = CubeObservationSpace(cube)
        self.reward_range = -1,1
    
    def _step(self):
        
    def _reset(self):
    
    def _render(self):
    
    def _close(self):
    
    def _seed(self):
        
class CubeActionSpace(Space):
    
    def __init__(self,cube):
        self.cube = cube
        
    def sample(self):
        self.cube.random_rotate()
        
    def contains(self,x):
        
class CubeObservationSpace(Space):
    
    def __init__(self,cube):
        self.cube = cube
        
    def sample(self):
        
    def contains(self,x):