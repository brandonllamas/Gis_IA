
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import fiona
import rasterio
from rasterio.plot import show
from skimage.feature import match_template
import numpy as np
from PIL import Image
import cv2
from osgeo import gdal
import numpy as np
import imutils
import os

class GisIAOpenCV():
    
    def __init__(self):
        self.rasterPath = ''
        self.shapePath = ''
        self.pointRatio = 25
        self.surveyRowCol = []
        self.selectedBand = ''
        self.matchXYList = []
        self.templateBandList = []
        
    def trainPoint(self,pathPointh):
        pointDat = fiona.pop(pathPointh)
        self.PointData = pointDat
        
    def openRaster(self,rasterPath):
        rasterRead = rasterio.open(rasterPath, dtype='uint16')
        
        print("Info Raster")
        print("Bands Rasters {}".format(rasterRead.crs))
        print("Numero de bandas {}".format(rasterRead.count))
        # print(rasterRead.read(2))
        # print("Color de bandas {}".format(rasterRead.colorinterp))
        
        self.rasterPath2 = rasterPath

    def openPoint(self,pointPath):
        pointDat= fiona.open(pointPath)
        
        # print("Point data => {}".format(pointDat.crs))
        
        self.pointData = pointDat 
              
    def getExamplesImage(self):
        rasterRead = rasterio.open(self.rasterPath2)
        greenBand = rasterRead.read(3)
        g_image = gdal.Open(self.rasterPath2)
        a_image = g_image.ReadAsArray()
        s_image = np.dstack((a_image[0],a_image[1],a_image[2]))
        # plt.imshow(s_image) # show image in matplotlib (no need for color swap)
        s_image = cv2.cvtColor(s_image,cv2.COLOR_RGB2BGR) # colorswap for cv
        image = cv2.imread(self.rasterPath2)
        # cv2.imwrite("objeto_{}.jpg".format(1),s_image)
        
        for index,values in enumerate(self.pointData):
            if values['geometry']!= None:
                pt = values['geometry']['coordinates']
                x = pt[0]
                y = pt[1]
                # tomamos el px del raster
                row,col = rasterRead.index(x,y)
                # lo pasamos
                self.surveyRowCol.append([row,col])
                
        for index,item in enumerate(self.surveyRowCol):
            
            # tomamos la posicion en px
            # print(item)
            row = item[0]
            rowSun = row+self.pointRatio
            rowRest = row-self.pointRatio
            
            col = item[1]
            
            colRest = col-self.pointRatio
            colSum = col+self.pointRatio
            
            # print(image.shape)
            imgAux = image.copy()
            
            objetoTest = imgAux[rowRest:rowSun, colRest:colSum]
            
            # objetoTest = imutils.resize(objetoTest,width=40)
            
            cv2.imwrite("image/method2/p/objeto_{}.jpg".format(index+1),objetoTest)
            print("imagen almacenada image/method2/p/objeto_{}.jpg".format(index+1))
            
               
        
                             