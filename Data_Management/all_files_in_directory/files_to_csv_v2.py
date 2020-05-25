"""
Script to unite txt docs into one csv file + additional conditions/filters
* does not concat text from multiple docs into one observation
"""

import glob, os
import fnmatch
import pandas as pd

rootdir = '.'
outdir = '.'
outfile = '/concat_4138.csv'


for root, dirs, files in os.walk(rootdir):
    contents = str()
    data = list()
    for dir in dirs:
        col_names = ['folder','doc','text']
        df = pd.DataFrame(columns = col_names)

    for file in files:
        filepath = root + os.sep + file
        if filepath.endswith(".txt"):
            # print (filepath)
            with open(filepath, 'r') as f:
                line = f.read()
                docname = file.split('.')[0]
                # if ('Office Depot' or 'office depot' or 'centralized claim intake sheet' or 'Centralized Claim Intake Sheet') in line:
                #     continue

                folder = root[2:]
                if len(folder) > 1:
                    if ('office depot') in line.lower():
                        df = df.append(dict({'folder': folder, 'doc': 'Office Depot cover sheet','text': line}, index=[0]), ignore_index=True)
                    if ('centralized claim intake sheet' or 'centralized intake coversheet') in line.lower():
                        df = df.append(dict({'folder': folder, 'doc': 'VA cover sheet','text': line}, index=[0]), ignore_index=True)
                    else:
                        df = df.append(dict({'folder': folder, 'doc': docname,'text': line}, index=[0]), ignore_index=True)




    # print('folder:',folder)
    # print('contents!! ',contents)
    # if len(folder) > 1:
    #     df = df.append(dict({'folder': folder, 'text': contents}, index=[0]), ignore_index=True)

df.to_csv(outdir+outfile, index=False)

print()
print(df)
print()
