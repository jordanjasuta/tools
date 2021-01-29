# tools
Living repo to stash useful tools for coding tasks that are bound to come up again

Contents include:
* __Data_Management__
    * __all_files_in_directory__ (scripts that operate on all the files in a root directory, subdirectories, etc for data processing and concatenation)
      * `change_file_type.py`  
      * `files_to_csv_v1.py`  (concats all files in subdirectory)
      * `files_to_csv_v2.py`  (includes some filters)
    * `API_call_json.py`
    * `SQL_data_pull.py`
    * `auto_update_test.py`
    * `unmelt_data.py`  (easier for human review)
* __GIS__
    * __GIS_python__
      * `shapefile_basics.py`  (geopandas + matplotlib)
      * `satellite_imagery_api.ipynb`  (rasterio + geopandas + gdal + geojson + shapely)
      * `basic_vessel_analytics.ipynb`  (plotly + mapbox)
    * __GIS_R__
      * `mapadecalor.Rmd`  (shapefiles, spatial join (polygons + points ), heatmap, natural breaks)
      * `datosgeoespaciales.Rmd`  (raster data, [rayshader](https://www.rayshader.com/) )
* __NLP__
    * `w2v_GloVe_LR.py`  (LR classification with gensim word2vec + GloVe basemodel)
    * `w2v_GloVe_MLPclf.py`  (MLP classification with gensim word2vec + GloVe basemodel)
    * `doc2vec_LR.py`  (LR classification with gensim doc2vec model)
    * `date_feature_extractor.py`  (datefinder for feature extraction)


_Note: if .ipynb file fails to load due to gh backend issues, use gh notebook url to view from [https://nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)_
