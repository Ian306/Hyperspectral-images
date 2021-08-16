import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from spectral import *
from skimage.exposure import histogram
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage.segmentation import watershed

%matplotlib qt5

cubo = open_image('C:/Users/iansa/Downloads/guayabas-20210713T013356Z-001/guayabas/GUAYABAS_Guayaba_GD_3_2021-07-08_18-48-31.hdr')

print(cubo.bands.centers[100])

#224 bandas 

#%% Visualizar bandas 
ncanal = 88 #706.62 nm
canal = cubo[:,:,ncanal]
plt.figure(1)
plt.clf()
plt.imshow(canal,cmap='jet')
plt.title('Imagen del canal ' + str(cubo.bands.centers[ncanal]) + ' nm '  )
plt.colorbar()
plt.show

#%% Normalized water difference index 
c1 = 180 #NIR 750 - 900nm 
c2 = 220 #SWIR 900 - 1000nm

Canal1 = np.float64(np.squeeze(cubo[:,:,c1]))    
Canal2 = np.float64(np.squeeze(cubo[:,:,c2]))  

num = Canal1 - Canal2 
den = Canal1 + Canal2

ratio = num / den

plt.figure(2)
plt.clf()
plt.imshow(ratio,cmap='jet')
plt.title('Normalized water difference index')
plt.colorbar()
plt.show()

#%% Perfil espectral 
# A(485, 600)

pA = np.squeeze(cubo[485, 600, :])

plt.figure(3)
plt.clf()
plt.plot(pA, '-r*')
plt.title('Perfil espectral del marcador A')
plt.show()

Resultado = np.zeros((ratio.shape[0], ratio.shape[1]), dtype = float)

for ren in range(ratio.shape[0]):
    for col in range(ratio.shape[1]):
        p = np.squeeze(cubo[ren, col, :])
        
        d = np.sqrt(np.sum(np.power(pA - p, 2)))

        Resultado[ren, col] = d
 
plt.figure(4)
plt.clf()
plt.imshow(Resultado,cmap='jet')
plt.title('Distancia euclidiana con respecto al perfil A')
plt.colorbar()
plt.show 


#%% k - means clustering 
# image = 706.62nm , clusters = 14, max_iterations = 20

(m, c) = kmeans(cubo, 20, 20)

plt.figure(5)
plt.clf()
plt.imshow(m,cmap='jet')
plt.title("k - means clustering")
plt.colorbar()
plt.show

#%% Principal components analysis

pc = principal_components(np.squeeze(canal))
v = imshow(pc.cov)
print(pc.eigenvalues)

#%% Mask Segmentation 

img = canal
mask = img < 600
img[mask] = 255
plt.figure(7)
plt.clf()
plt.imshow(img, cmap='jet')
plt.title("Mask segmentation")
plt.colorbar()
plt.show

#%% Edge-based segmentation

img2 = np.squeeze(canal)
edges = canny(img2/255.)
fill = ndi.binary_fill_holes(edges)
plt.figure(8)
plt.clf()
plt.imshow(fill)
plt.title("Edge based segmentation")
plt.colorbar()
plt.show

#%% Region based segmentation

elevation_map = sobel(img2)

markers = np.zeros_like(img2)
markers[img2 < 1500] = 1
markers[img2 > 3000] = 2

segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled, _ = ndi.label(segmentation)
plt.figure(9)
plt.clf()
plt.imshow(labeled)
plt.title("Region based segmentation")
plt.colorbar()
plt.show

#%% Perfil espectral de regiones 
# A(485, 600)

pA = np.squeeze(cubo[:, :, 116])

plt.figure(3)
plt.clf()
plt.plot(pA, '-r*')
plt.title('Perfil espectral del marcador A')
plt.show()

#%% Histogramas 

hist, hist_centers = histogram(canal)
plt.plot(hist)
