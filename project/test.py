import sqlite3
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

import pandas as pd
raw_data = {'col0': [1, 2, 3, 4], 'col1': [10, 20, 30, 40], 'col2':[100, 200, 300, 400]}
df = pd.DataFrame(raw_data)
print( df )



df.to_sql('test', conn ,if_exists='append' )  #test 테이블이 이미 존재하면 insert만,  없으면 새로 test 테이블 생성