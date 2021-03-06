import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import copy

class RubiksCube:
    facesdict = dict([("F",0),("U",1),("D",2),("L",3),("R",4),("B",5)])
   
    def __init__(self,n=3,state = None):
        '''
        Intialises a Rubik's cube object

        n: Dimension of the n x n x n cube
        state: An optional initial cube state. A cube state contains six boolean arrays representing the colours of each index.
        '''
        self.n=n
        self.cube_indices = np.arange(6*n**2).reshape((6,n,n))
        self.solved_cube = np.zeros((6,n**2*6),dtype=bool)
        
        for i,row in enumerate(self.solved_cube):
            self.solved_cube[i,i*n**2:(i+1)*n**2] = True
        if state is None:
            self._cube = self._initialize()
        else:
            self._cube = state
        self.face_array = self.build_face_array(n)
        self.z_slice = self.build_z_slice(n)
        self.move_list = []
        self.rots=self.rotations_list()
        
    def rotations_list(self):
        '''
        Creates a list of tuples which represents each possible move
        '''
        rots = []
        for row_index in range(self.n):
            for col_index in range(self.n):
                for direction in [-1,1]:
                    rots.append((row_index,col_index,direction))
        return rots

    def build_face_array(self,n):
        '''
        Creates an array which specifies the index of the fact 
        '''
        face_array = np.zeros((n,3),dtype=int)
        face_array.fill(-1)
        face_array[0,0] = 3
        face_array[n-1,0] = 4
        face_array[0,1] = 1
        face_array[n-1,1]= 2
        face_array[0,2] = 0
        face_array[n-1,2] = 5
        return face_array
    
    def build_z_slice(self,n):
        '''
        Builds an array which specifies the indices of elements on each face that rotate when rotating on the z-axis
        '''
        z_slice=[]
        for i in range(n):
            u_row = self.cube_indices[1,n-1-i,:]
            d_row = self.cube_indices[2,i,:]
            l_row = self.cube_indices[3,:,n-1-i]
            r_row = self.cube_indices[4,:,i]
            z_slice.append(np.array([u_row,l_row,d_row,r_row]))
        return np.array(z_slice)
    
    def cube_colours(self):
        '''
        Maps the boolean cube array onto on array of indices for each colour.
        cube: a reshaped cube to represent each face
        '''
        colour_map = np.zeros(6*self.n**2)
        for i,row in enumerate(self._cube):
            args = np.argwhere(row)
            colour_map[args] = i
        cube = colour_map.reshape((6,self.n,self.n))
        return cube
    
    def show_layout(self):
        '''
        Shows the layout of the cube
        '''
        cube = self.cube_colours()
        titles = ["F","B","U","D","L","R"]
        ix = [0,5,1,2,3,4]
        fig = plt.figure()
        self.patches = []
        for i in range(1,7):
            fig.add_subplot(3,2,i)
            patch=plt.imshow(cube[ix[i-1]], vmin=0, vmax=6, cmap='jet')
            plt.axis('off')
            plt.title(titles[i-1])
            self.patches.append(patch)
        return fig
      
    
    def _initialize(self,n=None):
        '''
        Initializes a cube.
        n: Number of random rotations of the cube
        '''
        if n is None:
            return self.solved_cube.copy()
        else:
            self._cube = self.solved_cube.copy()
            self.random_rotate(n)
            return self._cube
        
    def rotation(self,index):
        '''
        Performs a rotation using the index of the move in the rotation list
        '''
        self.rotate_cube(*self.rots[index])
        
    def rotate_cube(self,row_index,ax_index,direction):
        '''
        Rotates the cube along a specified row and axis index.
        '''
        face_ix = self.face_array[row_index,ax_index]
        rot = face_ix != -1
        if ax_index == 0:
            indices = self.cube_indices[:,:,row_index][[0,1,5,2]].flatten()
        elif ax_index == 1:
            indices = self.cube_indices[:,row_index,:][[0,3,5,4]].flatten()
        else:
            indices = self.z_slice[row_index].flatten()
        self._cube[:,indices] = np.roll(self._cube[:,indices],direction*self.n,axis=1)
        if rot:
            init = self.cube_indices[face_ix].flatten()
            fin = np.rot90(self.cube_indices[face_ix],direction).flatten()
            self._cube[:,fin] = self._cube[:,init] 
        self.move_list.append((row_index,ax_index,direction))
            
    def reset(self,seed):
        self.move_list = []
        self._cube = self._initialize(seed)
        return self._cube

    def score_similarity(self):
        return np.sum(self._cube.reshape(6,6,self.n**2),axis=2)/self.n**2

        
    def random_rotate(self,n=1,seed=None):
        '''
        Rotates the cube randomly. The cube can rotate either 90deg or 180deg
        '''
        if seed:
            np.random.RandomState(seed)
        ixs_ = np.random.choice(self.n,n)
        axis_ = np.random.choice(self.n,n)
        dirs_ = np.random.choice((-1,1),n)
        for x,y,z in zip(ixs_,axis_,dirs_):
            self.rotate_cube(x,y,z)
