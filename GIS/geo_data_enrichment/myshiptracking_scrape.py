#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import requests
import urllib.request
import time
import pandas as pd
from bs4 import BeautifulSoup
import os
import string
from datetime import datetime


class ScrapeVessels:
    """ scrape vessel data by MMSI
    """
    #
    def __init__(self):
        self.var = ''



    def map_vt(self, input):
        mapped = input.strip()
        with open('vt_mapping.txt') as f:
            lines = f.read().splitlines()
            for line in lines:
                if input.strip() in line.split(','):
                    mapped = line.split(',')[0]
                else:
                    continue

        return(mapped)


    def map_vessel_data(self, data):
        data['vesseltype'] = data['vesseltype'].apply(lambda x: self.map_vt(x))
        return(data)


    def get_vessels(self, mmsilist):

        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
           'AppleWebKit/537.36 (KHTML, like Gecko) '\
           'Chrome/75.0.3770.80 Safari/537.36'}

        vessel_type_dict = dict()
        mmsi_list = list()
        vesseltype_list = list()
        vesselflag_list = list()
        vesselname_list = list()
        mmsi_list = list()

        df = pd.DataFrame(columns=['mmsi', 'vesselname', 'vesseltype', 'vesselflag'])
        # print(df)

        # for letter in string.ascii_uppercase:
        for mmsi in mmsilist:

            mmsi_list.append(mmsi)
            url = "https://www.myshiptracking.com/vessels/mmsi-"+str(mmsi)
            # print(url)
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            # print(soup)
            if len(soup.find_all('tr')) > 0:
                for tr in soup.find_all('tr'):
                    # print(tr.text)
                    if 'name' in tr.text.lower():
                        tds = tr.find_all('td')
                        vesselname_list.append(tds[1].text)
                        # print(tds[1].text)
                    if 'type' in tr.text.lower():
                        tds = tr.find_all('td')
                        vesseltype_list.append(tds[1].text)
                    if 'flag' in tr.text.lower():
                        tds = tr.find_all('td')
                        vesselflag_list.append(tds[1].text.strip())
            else:
                break

        vt_df = pd.DataFrame(list(zip(mmsi_list, vesselname_list, vesseltype_list, vesselflag_list)),
                columns=['mmsi', 'vesselname', 'vesseltype', 'vesselflag'])
        df = df.append(vt_df)
        # map ship type categories to most basic type
        # df = self.map_vessel_data(df)

        return(df)




if __name__ == '__main__':

    SV = ScrapeVessels()
    data = SV.get_vessels([477141400, 366646910, 366062000])

    # data.to_csv('vessel_data.csv', index=False)

    print(data.head(10))





#
