# Machine-Learning-A-Z
Exploring Machine Learning from A to Z

Please note that time is in seconds and convert to hours to apply HourlyRate.

fpathdata="D:/gg0094889/TECHM/Project/AT&T/DLE/PyOmo/Data/Data/"
fname1='belgium-n100-k10.csv'
fname2='belgium-road-km-n100-k10.csv'
fname3='belgium-road-time-n100-k10.csv'
fname4='belgium-tw-d3-n100-k10.vrp'


df1=pd.read_csv(fpathdata+fname1)
df2=pd.read_csv(fpathdata+fname2)
df3=pd.read_csv(fpathdata+fname3)
df4=pd.read_csv(fpathdata+fname4)

dfLatLong = df1.iloc[7:107,0:3]
dfDemand  = df1.iloc[108:208,0:3]
dfkm      = df2.iloc[110:210,:]
dftime    = df3.iloc[110:210,:]
dfLatLong.columns=['LocId','Lat','Long']

dfkm.columns=dfLatLong['LocId'].unique()
dfkm.set_index(dfLatLong['LocId'].unique(),inplace=True)
dftime.columns=dfLatLong['LocId'].unique()
dftime.set_index(dfLatLong['LocId'].unique(),inplace=True)

dfkm=dfkm.astype(float)
dftime=dftime.astype(float)

dfDemand.columns=['LocId','Demand','Name']

print(df1.shape,df2.shape,df3.shape)


n1 = 108
n2 = n1+100
df4.iloc[n1:n2,:]
dfDem=df4.iloc[n1:n2,0].str.split(' ',expand=True)
dfDem.columns=["LocId","Demand","StartTime","EndTime","Capacity"]
dfDem

lstDemand = sorted(map(int,dfDemand['LocId'].unique()))
lstDem    = sorted(map(int,dfDem['LocId'].unique()))

dLocIdMap = dict(zip(lstDemand,lstDem))


dfkm.columns   = lstDem
dfkm.index     = lstDem
dftime.columns = lstDem
dftime.index   = lstDem
dfkm
