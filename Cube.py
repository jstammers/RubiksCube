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
    
    def rotate_cube(self,face,n):
        if face not in self.facesdict.keys():
            raise "Incorrect Face Specified"
        indices = self.rotation_indices(face)
        if n == 1:
            ix = [1,2,3,0]
            k = 1
        elif n == -1:
            ix = [3,0,1,2]
            k=3
        sides = self.get_adjacent_squares(indices)
        self.cube[self.facesdict[face]]=np.rot90(self.cube[self.facesdict[face]],k=k)
        self.set_adjacent_sqaures(indices,sides[np.ix_(ix)])
        
        
    def rotation_indices(self,face):
        if face == "F":
            sides = [("U",0,1),("R",0,1),("D",self.n-1,1),("L",0,1)]
        elif face == "U":
            sides = [("B",0,1),("R",0,0),("F",self.n-1,1),("L",self.n-1,0)]
        elif face == "D":
            sides = [("F",0,1),("R",self.n-1,0),("B",self.n-1,1),("L",0,0)]
        elif face == "L":
            sides = [("U",0,0),("F",0,0),("D",0,0),("B",0,0)]
        elif face == "R":
            sides =  [("U",self.n-1,0),("B",self.n-1,0),("D",self.n-1,0),("F",self.n-1,0)]
        else:
            sides = [("U",self.n-1,1),("R",self.n-1,1),("D",0,1),("L",self.n-1,1)]
        return sides
    
    def get_adjacent_squares(self,indices):
        '''
        Returns the squares that are adjacent to a given face. The faces are defined in a clockwise order, starting from the face above the given one. Each side is the index of the required slice, along with the axis
        '''
        l = []
        for face,ix,ax in indices:
            if ax == 1:
                l.append(self.cube[self.facesdict[face],:,ix])
            else:
                l.append(self.cube[self.facesdict[face],ix])
        return np.array(l)
    
    def set_adjacent_sqaures(self,indices,values):
        for i,(face,ix,ax) in enumerate(indices):
            if ax == 1:
                self.cube[self.facesdict[face],:,ix] = values[i]
            else:
                self.cube[self.facesdict[face],ix] = values[i]
                
    def score_similarity(self):
        a1 = self.cube.ravel()
        a2 = self.solved_cube.ravel()
        return sum(a1==a2)