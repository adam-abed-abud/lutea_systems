import model
import NOC
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import cloudsProcessor
import numpy as np

sourceDirName("./Data/")
sourceLabels("./Data.csv")

temp_PC = NOC.main.getPointsOnly(sourceDirName) #NOC
temp_PM = model.flowMapGenerate(temp_PC) #Model
temp_data = cloudsProcessor.prepareSTL(temp_PC, temp_PM, np.genfromtxt(sourceLabels, delimiter=','))
temp_mesh = mesh.Mesh(temp_data)
temp_mesh.save('outPut.stl')
