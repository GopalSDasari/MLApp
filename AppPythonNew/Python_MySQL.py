import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import configparser

PATH = os.getcwd()

config = configparser.ConfigParser()
config.read(PATH+'/conf/config.ini')

host        = config['MySQL']['host']
port        = config['MySQL']['port']
user        = config['MySQL']['user']
password    = config['MySQL']['password']
db          = config['MySQL']['db']

connector = 'mysql+mysqlconnector://' + user + ':' + password + '@' + host + ':' + port + '/' + db

bank = pd.read_sql("select * from bank", con=connector)

print(bank.head(4))

bank.replace(to_replace=['unknown'], value=np.nan, inplace=True)

print(bank.head(4))

# customer_no is not of much value to dropping it
bank = bank.drop(['customer_no'], axis=1)

print(bank.head(4))
print(bank.tail(4))
bank.dropna(subset = ['job', 'marital', 'eduation', 'housing', 'loan'], inplace = True)
Imp = SimpleImputer(strategy = 'most_frequent')
Imp.fit(bank)
Impbank = Imp.transform(bank)
bank = pd.DataFrame(Impbank, columns = bank.columns).astype(bank.dtypes.to_dict())
