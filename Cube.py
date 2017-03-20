import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
class RubiksCube:
    
    def __init__(self,n=3):
        self.n=n
        self.facesdict = dict([("F",0),("U",1),("D",2),("L",3),("R",4),("B",5)])
        self.solved_cube = np.array([np.full((3,3),s) for s in [1,2,3,4,5,6.5]],dtype=np.byte)
        self.cube = self._initialize(n)
       
    def show_layout(self):
        grid = np.zeros((9,12))
        grid[3:6,0:3] = self.cube[0]
        grid[0:3,3:6] = self.cube[3]
        grid[3:6,3:6] = self.cube[1]
        grid[6:9,3:6] = self.cube[4]
        grid[3:6,6:9] = self.cube[5]
        grid[3:6,9:12] = self.cube[2]

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
            sides = np.vstack(self.cube[ix]).T
            sides[index] = np.roll(sides[index],d*3)
            self.cube[ix] = np.vsplit(sides.T,4)
        elif axis == 1:
            if index == 0:
                face_ix = 3
                rot = True
            elif index == self.n-1:
                face_ix = 4
            #vertical
            ix = [0,1,5,2]   
            sides = np.hstack(self.cube[ix])
            sides[index] = np.roll(sides[index],d*3)
            self.cube[ix] = np.hsplit(sides,4)
            
        if rot: ar[face_ix] = np.rot90(slef.cube[face_ix],k=d)
        
                
    def score_similarity(self):
        a1 = self.cube.ravel()
        a2 = self.solved_cube.ravel()
        return sum(a1==a2)