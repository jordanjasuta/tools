"""
Simple API call (get)
"""

import requests
import pandas as pd
from pandas.io.json import json_normalize

adm1_url = "https://services.arcgis.com/5T5nSi527N4F7luB/arcgis/rest/services/GLOBAL_ADM1_P_API/FeatureServer/2/query?f=json&where=1%3D1&returnGeodetic=false&outFields=&returnGeometry=false&returnQueryGeometry=false&returnExceededLimitFeatures=true&outFields=WHO_REGION%2C+ADM0_CODE+AS+ISO3_CODE%2C+ADM0_NAME+AS+PLACE%2C+ADM1_CODE+AS+ADM1_ISOCODE%2C+ADM1_NAME+AS+ADM1%2C+WHO_SUBREGION%2C+GlobalID+AS+GUID_ADM1%2C+CENTER_LON+AS+LONGITUDE%2C+CENTER_LAT+AS+LATITUDE&token=pk578261AmKWp1g6HiWW-Jl2a-p-Q_RhpSdbSejDAV-dmv9s36DMd90YOXUXZOtbWjwpqUyZxW0t2mrnNChKKRjQMol4dChsREExs6kYBIu2wF7lQ_eah9VVl98MOK9ktRYHLCqQ_BX4jQLadxamFDmUcFJ3G74aw3IfKZ5CvMV_vN3JU-9JNP1GcBfSiQHeuTeVPP-yLU9PmrEnNwGsRuIu-WQSuj0DCmU4NBiXpWM"

response = requests.get(adm1_url)
adm1_json = response.json()
# data = pd.DataFrame(adm1_json.items(), columns=['key', 'fields'])
data = json_normalize(adm1_json, record_path="features")

print(data.shape)
print(data.head())
