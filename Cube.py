import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import copy
class RubiksCube:
    facesdict = dict([("F",0),("U",1),("D",2),("L",3),("R",4),("B",5)])
        
    def __init__(self,n=3,state = None):
        self.n=n
        self.solved_cube = np.array([np.full((self.n,self.n),s) for s in [1,2,3,4,5,6]])
        if state is None:
            self._cube = self._initialize(n)
        else:
            self._cube = state
        self.move_list = []
       
    def show_layout(self):
        #TODO: Update this to work with any size cube
        grid = np.zeros((9,12))
        grid[3:6,0:3] = self._cube[0]
        grid[0:3,3:6] = self._cube[3]
        grid[3:6,3:6] = self._cube[1]
        grid[6:9,3:6] = self._cube[4]
        grid[3:6,6:9] = self._cube[5]
        grid[3:6,9:12] = self._cube[2]

        return plt.imshow(grid,cmap=npimg.cm.jet)
    
    def _initialize(self,n):
        return self.solved_cube.copy()
    
    def rotate_cube(self,index,axis,d):
        rot = False
        if axis == 0:
            if index == 0:
                face_ix = 1
                rot = True
            elif index == self.n-1:
                face_ix = 2
                rot = True
            #horizontal
            ix = [0,3,5,4]
            sides = np.vstack(self._cube[ix]).T
            sides[index] = np.roll(sides[index],d*3)
            self._cube[ix] = np.vsplit(sides.T,4)
        elif axis == 1:
            if index == 0:
                face_ix = 3
                rot = True
            elif index == self.n-1:
                face_ix = 4
            #vertical
            ix = [0,1,5,2]   
            sides = np.hstack(self._cube[ix])
            sides[index] = np.roll(sides[index],d*3)
            self._cube[ix] = np.hsplit(sides,4)
            
        if rot: self._cube[face_ix] = np.rot90(self._cube[face_ix],k=d)
        
                
    def score_new_move(self,moves):
        new_cube = RubiksCube(self.n,self._cube)
        new_cube.rotate_cube(*moves)
        return new_cube.score_similarity()

    def score_similarity(self):
        a1 = self._cube.ravel()
        a2 = self.solved_cube.ravel()
        return(sum(a1==a2))

    def random_rotate(self,n=1,seed=None):
        '''
        Rotates the cube randomly. The cube can rotate either 90deg or 180deg
        '''
        if seed:
            np.random.RandomState(seed)
        ixs_ = np.random.choice(self.n,n)
        axis_ = np.random.choice((0,1),n)
        dirs_ = np.random.choice((-2,-1,1,2),n)
        for x,y,z in zip(ixs_,axis_,dirs_):
            self.rotate_cube(x,y,z)

    def solver_func(self,Solver):
        '''
        Attempts to solve the cube with a given policy function.

        policy: a function which given the current state, returns the next move in the form index,axis,d
        '''
        (ix,ax,d),score = Solver.policy(self)
        if score > 0:
            self.move_list.append((ix,ax,d))
            self.rotate_cube(ix,ax,d)
            return False
        else:
            return True