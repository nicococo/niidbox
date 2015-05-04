import numpy as np
import Image
from matplotlib.pyplot import imshow

# Define a color table
colorTable = (
        (0,0,143,255),     (0,0,159,255),     (0,0,175,255),     (0,0,191,255), 
        (0,0,207,255),     (0,0,223,255),     (0,0,239,255),     (0,0,255,255), 
       (0,16,255,255),    (0,32,255,255),    (0,48,255,255),    (0,64,255,255),
       (0,80,255,255),    (0,96,255,255),   (0,112,255,255),   (0,128,255,255),
      (0,143,255,255),   (0,159,255,255),   (0,175,255,255),   (0,191,255,255),
      (0,207,255,255),   (0,223,255,255),   (0,239,255,255),   (0,255,255,255), 
     (16,255,255,255),  (32,255,239,255),  (48,255,223,255),  (64,255,207,255), 
     (80,255,191,255),  (96,255,175,255), (112,255,159,255), (128,255,143,255), 
    (143,255,128,255), (159,255,112,255),  (175,255,96,255),  (191,255,80,255), 
     (207,255,64,255),  (223,255,48,255),  (239,255,32,255),  (255,255,16,255),
      (255,255,0,255),   (255,239,0,255),   (255,223,0,255),   (255,207,0,255), 
      (255,191,0,255),   (255,175,0,255),   (255,159,0,255),   (255,143,0,255), 
      (255,128,0,255),   (255,112,0,255),    (255,96,0,255),    (255,80,0,255), 
       (255,64,0,255),    (255,48,0,255),    (255,32,0,255),    (255,16,0,255), 
        (255,0,0,255),     (239,0,0,255),     (223,0,0,255),     (207,0,0,255),
        (191,0,0,255),     (175,0,0,255),     (159,0,0,255),     (143,0,0,255)
)

# Create a Vol class
class Vol:
    'Common base class for all volumes'
    
    # Read data from file
    def read(self,fileName):
        # Open the file for reading only
        f = open(fileName, 'r')
    
        # Read the file header
        f.readline() # skip the first line
        t1, sizeX, sizeY, sizeZ, originX, originY, originZ, stepX, stepY, stepZ, rotation = (f.readline()).split(' ')
        f.close()
    
        # Convert strings variables to the correct format
        self.sizeX = int(sizeX)
        self.sizeY = int(sizeY)
        self.sizeZ = int(sizeZ)
        self.originX = float(originX)
        self.originY = float(originY)
        self.originZ = float(originZ)
        self.stepX = float(stepX)
        self.stepY = float(stepY)
        self.stepZ = float(stepZ)
        self.rotation = float(rotation)
    
        # Now read all data from file
        vol = np.loadtxt(fileName, skiprows=3)
        vol = vol.reshape(self.sizeZ,self.sizeY,self.sizeX)
        self.data = vol
        
        # Set the maximum and minimum voxel and world coordinates
        self.vMinX = 0
        self.vMaxX = self.sizeX-1
        self.vMinY = 0
        self.vMaxY = self.sizeY-1
        self.vMinZ = 0
        self.vMaxZ = self.sizeZ-1
        self.wminX = self.originX
        self.wMaxX = self.originX + (self.sizeX-1)*self.stepX
        self.wminY = self.originY
        self.wMaxY = self.originY + (self.sizeY-1)*self.stepY
        self.wminZ = self.originZ
        self.wMaxZ = self.originZ + (self.sizeZ-1)*self.stepZ
    

    # Load data to volume structure
    def load(self,sizeX, sizeY, sizeZ, originX, originY, originZ, 
             stepX, stepY, stepZ, rotation, data):
   
       # Convert variables to the correct format
        self.sizeX = int(sizeX)
        self.sizeY = int(sizeY)
        self.sizeZ = int(sizeZ)
        self.originX = float(originX)
        self.originY = float(originY)
        self.originZ = float(originZ)
        self.stepX = float(stepX)
        self.stepY = float(stepY)
        self.stepZ = float(stepZ)
        self.rotation = float(rotation)
        self.data = np.copy(data)
        
        # Set the maximum and minimum voxel and world coordinates
        self.vMinX = 0
        self.vMaxX = self.sizeX-1
        self.vMinY = 0
        self.vMaxY = self.sizeY-1
        self.vMinZ = 0
        self.vMaxZ = self.sizeZ-1
        self.wminX = self.originX
        self.wMaxX = self.originX + (self.sizeX-1)*self.stepX
        self.wminY = self.originY
        self.wMaxY = self.originY + (self.sizeY-1)*self.stepY
        self.wminZ = self.originZ
        self.wMaxZ = self.originZ + (self.sizeZ-1)*self.stepZ
    

    # Save volume on file
    def save(self, fileName, description): 
                      
        f = open(fileName, 'w')
        f.write(description)
        f.write("1 "+str(self.sizeX)+" "+str(self.sizeY)+" "+str(self.sizeZ)+" "+
                '%.6f'% self.originX+" "+'%.6f' % self.originY+" "+'%.6f' % self.originZ+
                " "+'%.6f' % self.stepX+" "+'%.6f' % self.stepY+" "+'%.6f' % self.stepZ+
                " "+'%.6f' % self.rotation+"\n")
        f.write('VValue\n')
        for z in range(self.sizeZ):
            for y in range(self.sizeY):
                for x in range(self.sizeX):
                    f.write(str(self.data[z,y,x])+"\n")
        f.close()
             
    # Generate a sub-volume and save it on file
    def saveSubVol(self, fileName, description, 
                   initX, initY, initZ, endX, endY, endZ):
    
        # Define voxel initial coordinates
        init_vx_X = int(round((initX-self.originX)/self.stepX))
        init_vx_Y = int(round((initY-self.originY)/self.stepY))
        init_vx_Z = int(round((initZ-self.originZ)/self.stepZ))
        dim_vx_X = abs(int(round((endX-initX+1.)/self.stepX)))
        dim_vx_Y = abs(int(round((endY-initY+1.)/self.stepY)))
        dim_vx_Z = abs(int(round((endZ-initZ+1.)/self.stepZ)))
    
        # Generate volumetric data
        subVol = np.zeros((dim_vx_Z,dim_vx_Y,dim_vx_X))
        for i in range(dim_vx_Z):
            for j in range(dim_vx_Y):
                for k in range(dim_vx_X):
                    subVol[i,j,k] = self.data[init_vx_Z+i,init_vx_Y+j,init_vx_X+k]
                
        # Save the sub-volume to fileName
        f = open(fileName, 'w')
        f.write(description)
        f.write("1 "+str(dim_vx_X)+" "+str(dim_vx_Y)+" "+str(dim_vx_Z)+" "+
                '%.6f' % initX+" "+'%.6f' % initY+" "+'%.6f' % initZ+" "+
                '%.6f' % self.stepX+" "+'%.6f' % self.stepY+" "+'%.6f' % self.stepZ+
                " "+'%.6f' % self.rotation+"\n")
        f.write('VValue\n')
        for z in range(dim_vx_Z):    
            for y in range(dim_vx_Y):
                for x in range(dim_vx_X):
                    f.write(str(subVol[z,y,x])+"\n")
        f.close()
 
        
    # Get the reservoir Z base coordinates
    def getResBase(self):
        refBase = np.zeros((self.sizeX,self.sizeY),dtype=int)
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for z in range(self.sizeZ-1,-1,-1):
                    if self.data[z,y,x] > -1e+30:
                        refBase[x][y] = z
                        break
        return refBase                

    
    # Get the reservoir Z top coordinates
    def getResTop(self):
        refTop = np.zeros((self.sizeX,self.sizeY),dtype=int)
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for z in range(self.sizeZ):
                    if self.data[z,y,x] > -1e+30:
                        refTop[x][y] = z
                        break
        return refTop
    
    
    # Plot a slice from the volume with d% distance from the bottom of the reservoir
    def plotSlice(self, d, color_table=colorTable):
    
        # Get the maximum and minimum values, excluding the "NaN value" (-1e+30)
        aux = np.sort(np.unique(self.data))
        minValue = aux[0]
        if minValue == -1e+30:
            minValue = aux[1]
        maxValue = aux[-1]

        # Define min and max color indexes
        minIndex = 0
        maxIndex = len(color_table)-1
    
        # Generate the color index volume
        vol_color = np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=int)
        vol_color = np.round((self.data-minValue)/(maxValue-minValue)*(maxIndex-minIndex)+minIndex).astype(int)
        
        # Get the reservoir z top coordinates
        refZT = self.getResTop()
                     
        # Get the reservoir z base coordinates
        refZB = self.getResBase()

        # Create and display image
        img = Image.new( 'RGBA', (self.sizeX,self.sizeY), "white") # create a new white image
        pixels = img.load() # create the pixel map

        for i in range(img.size[0]):  # for every pixel
            for j in range(img.size[1]):
                l = refZB[i][j]-refZT[i][j]
                z =refZB[i][j]-int(round(d*l))
                pixels[i,j] = (color_table[vol_color[z,j,i]][0], 
                               color_table[vol_color[z,j,i]][1], 
                               color_table[vol_color[z,j,i]][2], 
                               color_table[vol_color[z,j,i]][3])
        imshow(np.array(img), origin='lower')
        
        
    # Plot a slice from the volume
    def plotSlice(self, sliceNumber, color_table=colorTable):
    
        # Get the maximum and minimum values, excluding the "NaN value" (-1e+30)
        aux = np.sort(np.unique(self.data))
        minValue = aux[0]
        if minValue == -1e+30:
            minValue = aux[1]
        maxValue = aux[-1]

        # Define min and max color indexes
        minIndex = 0
        maxIndex = len(color_table)-1
    
        # Generate the color index volume
        vol_color = np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=int)
        vol_color = np.round((self.data-minValue)/(maxValue-minValue)*(maxIndex-minIndex)+
                    minIndex).astype(int)
        
         # Create and display image
        img = Image.new( 'RGBA', (self.sizeX,self.sizeY), "white") # create a new white image
        pixels = img.load() # create the pixel map

        for i in range(img.size[0]):  # for every pixel
            for j in range(img.size[1]):
                pixels[i,j] = (color_table[vol_color[sliceNumber,j,i]][0], 
                               color_table[vol_color[sliceNumber,j,i]][1], 
                               color_table[vol_color[sliceNumber,j,i]][2], 
                               color_table[vol_color[sliceNumber,j,i]][3])
        imshow(np.array(img), origin='lower')        
