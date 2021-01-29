"""
Script to pull data from space track for final purpose of enriching TLE data.

WARNING: DO NOT RUN THIS SCRIPT TOO MANY TIMES IN A SINGLE DAY.
         THIS API IS SUBJECT TO LIMITS AND BLOCKING.
"""

import pandas as pd
# import numpy as np
from spacetrack import SpaceTrackClient


class SpaceTrackPull():
    def __init__(self):
        with open("st_authentication.txt","r") as f:
            lines=f.readlines()
        self.user = lines[0].strip()
        self.password = lines[1].strip()

    def pull_catalogue(self):
        st = SpaceTrackClient(identity=self.user, password=self.password)
        st_data = st.satcat()
        # st_data = st.satcat(norad_cat_id=ids)

        intldes = list()  # do we need this?
        norad_cat_id = list()
        object_type = list()
        country = list()
        decay = list()
        launch_year = list()

        # dict to df (might change to json later)
        for dict in st_data:
            print(dict['INTLDES'])
            intldes.append(dict['INTLDES'])
            norad_cat_id.append(dict['NORAD_CAT_ID'])
            object_type.append(dict['OBJECT_TYPE'])
            country.append(dict['COUNTRY'])
            decay.append(dict['DECAY'])
            launch_year.append(dict['LAUNCH_YEAR'])

        st_df = pd.DataFrame(list(zip(intldes, norad_cat_id, object_type, country, decay, launch_year)),
                columns =['INTLDES', 'NORAD_CAT_ID', 'OBJECT_TYPE', 'COUNTRY', 'DECAY', 'LAUNCH_YEAR'])

        return(st_df)


    def query_satcat(self, satcat_path, ids):
        # load data
        satcat = pd.read_csv(satcat_path)
        # 'query' data
        results = satcat.loc[satcat['NORAD_CAT_ID'].isin(ids)]
        print()
        print(results.to_string(index=False))
        print(len(results), 'resutls found for', len(ids), 'norad cat ids')
        print()



if __name__ == '__main__':

    st = SpaceTrackPull()
    satcat_path = 'data/spacetrack_catalogue.csv'
    ids = [26619, 29, 27347, 11015, 120, 27421, 260, 4564, 2864, 25544, 29268, 33312, 7195,
           21286, 22328, 16615, 38119, 32768, 39487, 41857, 28082, 5187]

    # TO PULL DATA
    data = st.pull_catalogue()
    data.to_csv(data_output_path, index=False)

    # # TO QUERY DATA
    # st.query_satcat(satcat_path,ids)




    #
