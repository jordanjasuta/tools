import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os

def excel_to_txt(file):
    filename = os.path.splitext(os.path.basename(file))[0]
    df=pd.read_excel(file, sheet_name='Sheet1')
    df.to_csv((filename + '.txt'), header=None, index=None, sep=' ', mode='a')

cdir = "." # Current directory
xlsxcdir = [filename for filename in os.listdir(cdir) if filename.endswith(".xlsx")]

for f in xlsxcdir:
    excel_to_txt(f)
