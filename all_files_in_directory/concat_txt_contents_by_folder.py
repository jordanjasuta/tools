"""
Script to unite 4138 txt docs from 41 machine
"""

import glob, os
import pandas as pd

rootdir = '.'
outdir = '.'
outfile = '/concat_4138.csv'


for root, dirs, files in os.walk(rootdir):
    contents = str()
    data = list()
    for dir in dirs:
        col_names = ['folder','text']
        df = pd.DataFrame(columns = col_names)

    for file in files:
        filepath = root + os.sep + file
        if filepath.endswith(".txt"):
            # print (filepath)
            with open(filepath, 'r') as f:
                line = f.read()
                # print(dir)
                contents += '\n'
                contents += line
    folder = root[2:]
    # print('folder:',folder)
    # print('contents!! ',contents)
    if len(folder) > 1:
        df = df.append(dict({'folder': folder, 'text': contents}, index=[0]), ignore_index=True)

df.to_csv(outdir+outfile, index=False)

print()
print(df)
print()
