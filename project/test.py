import sqlite3
conn = sqlite3.connect('db.sqlite3')
#conn.execute("drop table Phrase")
c = conn.cursor()


import pandas as pd
#raw_data = {'col0': [1, 2, 3, 4], 'col1': [10, 20, 30, 40], 'col2':[100, 200, 300, 400]}
Phrase = pd.read_excel('Phrase_DB.xlsx', sheet_name='Phrase_DB', encoding='utf-8')
print( Phrase )



Phrase.to_sql('smile_phrase', conn ,if_exists='append', index=False)  #test 테이블이 이미 존재하면 insert만,  없으면 새로 test 테이블 생성P

#help(pd.Dataframe.to_sql())