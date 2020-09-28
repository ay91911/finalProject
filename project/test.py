
# *시스템명: 감성82
# *업무명칭 : 엑셀내용 -> 테이블 추가
# * File Name 	: test.py
# * Author 	    : 유아영
# * 생성일  	    : 2020.09.26
# * Description  	:
# 이거..... 애드라.. 내가 열심히 엑셀내용을 만들어진 테이블에 붙여넣을려고 했는데, 자꾸 에러가 뜨네?
# 그래서 그냥 이 코드 안쓰고, sqliteStudio에서 셀 추가해서 만들었음. 참고바람
# Phrase_DB엑셀 파일은 그래서 필요없음






import sqlite3
conn = sqlite3.connect('db.sqlite3')
#conn.execute("drop table Phrase")
c = conn.cursor()


import pandas as pd
#raw_data = {'col0': [1, 2, 3, 4], 'col1': [10, 20, 30, 40], 'col2':[100, 200, 300, 400]}
Phrase = pd.read_excel('Phrase_DB.xlsx', sheet_name='Phrase_DB', encoding='utf-8')
Phrase
print( Phrase )


#Phrase.to_sql('smile_PHRASE', conn ,if_exists='append')  #test 테이블이 이미 존재하면 insert만,  없으면 새로 test 테이블 생성P

Phrase.to_sql('smile_PHRASE', conn ,if_exists='append', index=False)  #test 테이블이 이미 존재하면 insert만,  없으면 새로 test 테이블 생성P

#help(pd.Dataframe.to_sql())