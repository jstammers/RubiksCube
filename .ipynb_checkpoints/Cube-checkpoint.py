import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import copy

class RubiksCube:
    facesdict = dict([("F",0),("U",1),("D",2),("L",3),("R",4),("B",5)])
   
    def __init__(self,n=3,state = None):
        self.n=n
        self.solved_cube = np.zeros((6,n**2*6),dtype=bool)
        self.cube_indices = np.arange(6*n**2).reshape((6,n,n))
        for i,row in enumerate(self.solved_cube):
            self.solved_cube[i,i*n**2:(i+1)*n**2] = True
        if state is None:
            self._cube = self._initialize()
        else:
            self._cube = state
        #Bulds an array to specify which face is rotated if an edge is rotated
        self.face_array = self.build_face_array(n)
        self.z_slice = self.build_z_slice(n)
        self.move_list = []
        self.rotations_list()
        
    def rotations_list(self):
        self.rots = []
        for i in range(self.n):
            for j in range(self.n):
                for k in [-1,1]:
                    self.rots.append((i,j,k))
        
    def build_face_array(self,n):
        face_array = np.zeros((n,n),dtype=int)
        face_array.fill(-1)
        face_array[0,0] = 3
        face_array[0,n-1] = 0
        face_array[0,1] = 1
        face_array[0,n-1]= 2
        face_array[n-1,0] = 4
        face_array[n-1,n-1] = 5
        return face_array
    
    def build_z_slice(self,n):
        b=[]
        for i in range(n):
            a1 = self.cube_indices[1,n-1-i,:]
            a2 = self.cube_indices[2,i,:]
            a3 = self.cube_indices[3,:,n-1-i]
            a4 = self.cube_indices[4,:,i]
            b.append(np.array([a1,a3,a2,a4]))
        return np.array(b)
    
    def cube_colours(self):
        colour_map = np.zeros(6*self.n**2)
        for i,row in enumerate(self._cube):
            args = np.argwhere(row)
            colour_map[args] = i
        cube = colour_map.reshape((6,self.n,self.n))
        return cube
    
    def show_layout(self,cube=None):
        if cube is None:
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
      
        
    def update_plot(self,frame):
        print(frame[0][0:3])
        for i in range(6):
            self.patches[i].set_data(frame[i])
        return self.patches
    
    def _initialize(self,n=None):
        if n is None:
            return self.solved_cube.copy()
        else:
            self._cube = self.solved_cube.copy()
            self.random_rotate(n)
            return self._cube
        
    def rotation(self,index):
        self.rotate_cube(*self.rots[index])
        
    def rotate_cube(self,index,axis,d):
        face_ix = self.face_array[index,axis]
        rot = face_ix != -1
        if axis == 0:
            indices = self.cube_indices[:,:,index][[0,1,5,2]].flatten()
        elif axis == 1:
            indices = self.cube_indices[:,index,:][[0,3,5,4]].flatten()
        else:
            indices = self.z_slice[index].flatten()
        self._cube[:,indices] = np.roll(self._cube[:,indices],d*self.n,axis=1)
        if rot:
            init = self.cube_indices[face_ix].flatten()
            fin = np.rot90(self.cube_indices[face_ix],d).flatten()
            self._cube[:,fin] = self._cube[:,init] 
        self.move_list.append((index,axis,d))
            
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
