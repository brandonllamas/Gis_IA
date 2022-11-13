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
        self.pointRatio = 25
        self.surveyRowCol = []
        self.selectedBand = ''
        self.matchXYList = []
        self.templateBandList = []
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
        
    def returnTiff(self):
        rasterRead = rasterio.open(self.rasterPath2)
        return rasterRead
    
    def openPoint(self,pointPath):
        pointDat= fiona.open(pointPath)
        
        # print("Point data => {}".format(pointDat.crs))
        
        self.pointData = pointDat
        
    def showFigWithPoint(self):
        fig,ax = plt.subplots(figsize=(10,10))
        rasterRead = rasterio.open(self.rasterPath2)
        # print(rasterRead)
        # print(self.pointData.crs)
        ax.axis('off')
        for point in self.pointData:
            if point['geometry']!= None:
                # print(point['geometry']['coordinates'])
                ax.scatter(point['geometry']['coordinates'][0],point['geometry']['coordinates'][1], c='orangered', s=100)
        show(rasterRead,ax=ax)
        plt.show()
    
    # ver puntos y las imagenes guias 
    def viewRasterPoint(self):
        # print(self.rasterPath2)
        rasterRead = rasterio.open(self.rasterPath2)
        greenBand = rasterRead.read(2)
        self.greenBand2 = greenBand
        # print(greenBand)
        #region recorrer puntos
        for index,values in enumerate(self.pointData):
            if values['geometry']!= None:
                pt = values['geometry']['coordinates']
                x = pt[0]
                y = pt[1]
                # tomamos el px del raster
                row,col = rasterRead.index(x,y)
                # lo pasamos
                self.surveyRowCol.append([row,col])
        #endregion
        
        fig,ax = plt.subplots(1,len(self.surveyRowCol),figsize=(20,5))
        
        # for index,item in enumerate(self.surveyRowCol):
        for index,item in enumerate(self.surveyRowCol):
            
            # tomamos la posicion en px
            # print(item)
            row = item[0]
            col = item[1]
            # print ("x => {};y => {}".format(row,col))
            # print(greenBand)
            # print(row,col)
            # imprimimos el aja la imaagen con la banda
            # show(greenBand)
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
    
    # obtener mas templates de referencias
    def getMoreTemplates(self):
        self.templateBandList = []
        rasterRead = rasterio.open(self.rasterPath2)
        greenBand = rasterRead.read(1)
        
        for rowCol in self.surveyRowCol:
            imageList = []
            # index pixeles
            row = rowCol[0]
            col = rowCol[1]
            # recortamos la imagen
            imageList.append(greenBand[row-self.pointRatio:row+self.pointRatio,col-self.pointRatio:col+self.pointRatio])
            # ahora colocamos mas grande imagen para que no se dañe
            templateRotate = greenBand[row - 2*self.pointRatio:row + 2*self.pointRatio,col - 2*self.pointRatio:col + 2*self.pointRatio ]
            # ahora creamos un rango de rotacion 
            rotationList = [i * 30 for i in range(1,4)]
            
            for rotation in rotationList:
                # tomamos la imagen
                rotatedRaw = Image.fromarray(templateRotate)
                # le metemos una rotacion
                rotatedImage = rotatedRaw.rotate(rotation)
                # pasamos el array y le ponemos el array normal
                imageList.append(np.asarray(rotatedImage)[self.pointRatio:-self.pointRatio,self.pointRatio:-self.pointRatio])
            
            # mostramos 
            # creamos una figura 12x12
            fig,ax = plt.subplots(1,len(imageList),figsize=(12,12))
            # lo reocrremos
            for index,item in enumerate(imageList):
                
                ax[index].imshow(imageList[index])
                ax[index].axis('off')
                ax[index].set_title(index+1)
            
            self.templateBandList += imageList
                
    def learnMethod1(self):
        rasterRead = rasterio.open(self.rasterPath2)
        greenBand = rasterRead.read(2)
        self.matchXYList = []
        # recorremos todas las coicidencias
        for index, templateBand in enumerate(self.templateBandList):
            # cada vez que van 10 aprendizaje lo muestra 
            if index%10 == 0:
                print("Hemos revisando {} figuras".format(index))
            
            # aplicamos el match template para que aprenda
            matchTemplate = match_template(greenBand,templateBand,pad_input=True)
            # filtramos el resultado a mayor de 0.996 
            matchTemplateFilter = np.where(matchTemplate>np.quantile(matchTemplate,0.9996))
            # recorro el xy y y lo meto en un array
            for item in zip(matchTemplateFilter[0],matchTemplateFilter[1]):
                x,y = rasterRead.xy(item[0],item[1])
                # lo guardamos
                self.matchXYList.append([x,y])
        
        # filtramos
        brc = Birch(branching_factor=10000,n_clusters=None,threshold=2e-5,compute_labels=True)
        brc.fit(self.matchXYList)
        # obtenemos los puntos centrales que encontramoes
        self.puntosFind = brc.subcluster_centers_
        
        # lo visualizamos
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.scatter(self.puntosFind[:,[0]],self.puntosFind[:,[1]],marker='o',color="orangered",s=100)
        show(rasterRead,ax=ax)
        plt.show()
        
    def exportPoint(self,nameArchive):
        np.savetxt("out/"+nameArchive,self.puntosFind,delimiter=",")