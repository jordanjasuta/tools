"""
unmelt data (easier for human review)
"""

book1 = pd.read_csv('Book1.csv', encoding='latin1')
print('length of original dataset: ', len(book1))

# keep only relevant field names
# book1 = book1[['Doc_ID','FIELD_NAME','FIELD_VALUE','CONFIDENCE']]
# book1 = book1[(book1['FIELD_NAME'] == 'VETERAN_BENEFICIARY_REMARKS') |
#             (book1['FIELD_NAME'] == 'VETERAN_BENEFICIARY_REMARKS_CLASSIFICATION') |
#             (book1['FIELD_NAME'] == 'VETERAN_BENEFICIARY_REMARKS_KEYWORDS')]

print(book1.head())

# concat field values with the same ID and field name, if applicable
book1['id_field'] = book1['Doc_ID'].map(str) + '-' + book1['FIELD_NAME'].map(str)
book1 = book1.fillna('').groupby('id_field')['FIELD_VALUE'].apply(' '.join)
book1 = pd.DataFrame(book1).reset_index()

print(book1.head())

#expand / reorder, if applicable
book1[['Doc_ID','FIELD_NAME']] = book1['id_field'].str.split('-',expand=True)
book1 = book1.drop(columns=['id_field'])
book1 = book1[['Doc_ID','FIELD_NAME','FIELD_VALUE']]

# unmelt
reshaped = book1.pivot(index='Doc_ID', columns='FIELD_NAME')['FIELD_VALUE'].reset_index()
reshaped.columns.name = None

print(reshaped.head())
print('length of new dataset: ', len(reshaped))

# save to csv
reshaped.to_csv('reshaped.csv')
