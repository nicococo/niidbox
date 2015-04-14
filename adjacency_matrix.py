import numpy as np

# Adjacency Matrix class
class AdjacencyMatrix:
    'Base class for adjacency matrices used in undirected simple graphs'
    
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n = self.nx*self.ny*self.nz
        self.a = np.zeros(self.n*(self.n+1)/2, dtype=bool) # alloc memory only for the upper diagonal matrix
                                                           # (it is symmetric)
        
    def location(self, x,y,z):
        return z*self.nx*self.ny + y*self.nx + x
    
    def mat2vec(self, li,lj):
        if (li <= lj):
            return li*self.n - (li-1)*li/2 + lj - li
        else:
            return lj*self.n - (lj-1)*lj/2 + li - lj

    def addEdge(self, xi,yi,zi, xj,yj,zj):
        li = self.location(xi,yi,zi)
        lj = self.location(xj,yj,zj)
        self.a[self.mat2vec(li,lj)] = True
    
    def removeEdge(self, xi,yi,zi, xj,yj,zj):
        li = self.location(xi,yi,zi)
        lj = self.location(xj,yj,zj)
        self.a[self.mat2vec(li,lj)] = False
        
    def hasEdge(self, xi,yi,zi, xj,yj,zj):
        li = self.location(xi,yi,zi)
        lj = self.location(xj,yj,zj)
        return self.a[self.mat2vec(li,lj)]
    
    def loc2xyz(self, loc):
        z = int(loc/(self.nx*self.ny))
        y = int((loc-z*self.nx*self.ny)/self.nx)
        x = int(loc - z*self.nx*self.ny - y*self.nx)
        return x,y,z
    
    def neighbors(self, x,y,z):
        li = self.location(x,y,z)
        v = []
        for lj in range(self.n):
            if self.a[self.mat2vec(li,lj)] == True:
                v.append(self.loc2xyz(lj))
        return v
    
    def neighborsLoc(self, li):
        v = []
        for lj in range(self.n):
            if self.a[self.mat2vec(li,lj)] == True:
                v.append(lj)
        return v    
