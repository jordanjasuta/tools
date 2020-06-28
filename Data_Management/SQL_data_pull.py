"""
SQL database pull
"""

import pandas as pd
import pyodbc


if __name__ == '__main__':
    info = {
     'server' : 'servername',    # info must be updated per project
     'database' : 'databasename',
     'username' : 'username',
     'password' : "passwordname"
     }

    conn = pyodbc.connect(driver='{ODBC Driver 17 for SQL Server}',
                                server=info['server'],
                               # port=1433,
                                database='database',
                                uid=info['username'],
                                pwd=info['password'])
  #  print(conn)

query = (
""" SELECT TOP (100) [UNIQUE_ID]
      ,[FIELD_NAME]
      ,[FIELD_VALUE]
      ,[EXTRACTION_TIMSTAMP]
  FROM [DATABASE].[DATATABLE]
  where ([FIELD_NAME] = 'somefield' or [FIELD_NAME] = 'otherfield')
  order by [UNIQUE_ID] desc, [FIELD_NAME] desc, [EXTRACTION_TIMSTAMP] desc """)


# make df from query results
raw = pd.read_sql(query, conn)

raw['TRUTH'] = raw['FIELD_VALUE']  # pull field values as truth
raw.loc[raw.FIELD_NAME != 'somefield', 'TRUTH'] = ""   # drow all values from truth except the final value POA code (all POA codes are final values, thus it does not need to be specified)

# repeat the final org value for all values with the same Doc_ID (this is currently done by order, which is not very stable)
groups = raw.groupby('Doc_ID')
res = []
for name, df in groups:
    x = list(set(df['TRUTH'].to_list()))
    if len(x) <2:
        temp = ['EMPTY' for i in range(len(df))]
        continue
    else:
        temp = [x[1] for i in range(len(df))]
        #codes.append(x[1])
    df['NEW_TRUTH'] = temp
    res.append(df)

total = pd.concat(res)

output = total[total['IS_FINAL_VALUE'] != 'Y']
cols = ['NEW_TRUTH','FIELD_VALUE']
output = output[cols]
output.columns = ['TRUTH','TEXT']

print(output.head())

output.to_csv('testdata.csv', sep=',', index=False)

print('test data has been saved to root directory')
