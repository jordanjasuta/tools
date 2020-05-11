### basic script to import shapefiles in Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon

sf_path = '../../../Downloads/Roads/Roads-polyline.shp'
sf = gpd.read_file(sf_path, encoding='utf-8')

print(sf.head())

stgo_shape = sf.to_crs({'init': 'epsg:4326'})
# stgo_shape.plot()

colors = {'primary':'1. primary', 'secondary':'2. secondary', 'tertiary':'3. tertiary', 'residential':'4. residential', 'unclassified':'5. unclassified'}

# ax = stgo_shape.plot(column='highway', cmap='magma', k=5, legend=True, figsize=(15,15))
ax = stgo_shape.plot(column=sf['highway'].apply(lambda x: colors[x]), cmap='magma', k=5, legend=True, figsize=(15,15))

plt.title('Roads in Quito');
# ax.set_axis_off()
ax.set_facecolor('#DCDCDC')
# plt.grid(b=True, color='#DCDCDC', linestyle='-')
plt.show()
