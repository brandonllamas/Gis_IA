from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import fiona
import rasterio
from rasterio.plot import show
from skimage.feature import match_template
import numpy as np
from PIL import Image

class cropMachine():
    
    # contructor
    def __init__(self):
        self.rasterPath = ''
        self.shapePath = ''
        self.pointRatio = 10
        self.surveyRowCol = []
        self.selectedBand = ''
        self.matchXYList = []
        # self.pointData = [];
        
    
    def openRaster(self,rasterPath):
        rasterRead = rasterio.open(rasterPath, dtype='uint16')
        
        print("Info Raster")
        print("Bands Rasters {}".format(rasterRead.crs))
        print("Numero de bandas {}".format(rasterRead.count))
        # print(rasterRead.read(2))
        # print("Color de bandas {}".format(rasterRead.colorinterp))
        
        self.rasterPath2 = rasterPath
        
    def openRasterW(self,raster):
        self.rasterF = raster
    
    def openPoint(self,pointPath):
        pointDat= fiona.open(pointPath)
        
        print("Point data => {}".format(pointDat.crs))
        
        self.pointData = pointDat
        
    def showFigWithPoint(self):
        fig,ax = plt.subplots(figsize=(10,10))
        rasterRead = rasterio.open(self.rasterPath2)
        # print(rasterRead)
        # print(self.pointData.crs)
        for point in self.pointData:
            if point['geometry']!= None:
                ax.scatter(point['geometry']['coordinates'][0],point['geometry']['coordinates'][1], c='orangered', s=100)
        show(rasterRead,ax=ax)
        plt.show()
    
    # ver puntos y las imagenes guias 
    def viewRasterPoint(self):
        print(self.rasterPath2)
        rasterRead = rasterio.open(self.rasterPath2)
        greenBand = rasterRead.read(2)
        print(greenBand)
        #region recorrer puntos
        for index,values in enumerate(self.pointData):
            if values['geometry']!= None:
                pt = values['geometry']['coordinates']
                x = pt[0]
                y = pt[1]
                
                row,col = rasterRead.index(x,y)
                self.surveyRowCol.append([x,y])
        #endregion
        
        fig,ax = plt.subplots(1,len(self.surveyRowCol),figsize=(20,5))
        
        # for index,item in enumerate(self.surveyRowCol):
        for index,item in enumerate(self.surveyRowCol):
            
            # tomamos la posicion en px
            row = item[0]
            col = item[1]
            # print(greenBand)
            # print(row,col)
            # imprimimos el aja la imaagen con la banda
            ax[index].imshow(greenBand)
            # y marcamos
            ax[index].plot(col,row,color="red",linestyle="dashed",marker="+",markerfacecolor="blue",markersize=8)
            # mani cogemos el col y le restamos el self.pointRatio pa que muestre solo la mata
            ax[index].set_xlim(col-self.pointRatio,col+self.pointRatio)
            # lo mosmi para esto
            ax[index].set_ylim(row-self.pointRatio,row+self.pointRatio)
            # quitamos los numeros de coordenada
            ax[index].axis('off')
            # cambiamos el titulo
            ax[index].set_title(index+1)
            # y asi mostramos la imagenes que buscaremos
            # if index == 5 : break
        
        
        