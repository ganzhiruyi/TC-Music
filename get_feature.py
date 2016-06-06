import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
PNG_DIR	= "image/statistics"
USER_WITH_ARTIST_FILE_PATH = "data/user_with_artist.csv"

def plot_image(aid,data):
	# use to plot the image of singer
	data.plot()
	path = PNG_DIR + str(aid) + ".png"
	plt.savefig(path)

def get_artist_data_to_file():
	# get the numbers of user action for each artist_id, and put them into id
	user_with_artist_data = pd.read_csv(USER_WITH_ARTIST_FILE_PATH, index_col=0)
	tmp = user_with_artist_data[['artist_id','action_type','Ds']]
	tmp['num']=1 # add a column 1 for groupby sum count.
	groups = tmp.groupby(['artist_id'])
	artist_ids = ""
	cnt_artist = 0
	for aid,group in groups:
		print aid
		artist_ids += aid + "\n"
		# print group.groupby(['Ds','action_type']).sum()
		data = group.groupby(['Ds','action_type']).sum().unstack().fillna(0)
		cnt_artist += 1
		data_path = "data/artist/"+str(cnt_artist)+".csv"
		data.to_csv(data_path, header=None)

	fp = open('data/artist/artist_id','w')
	fp.write(artist_ids)
	fp.close()

def get_artist_data_as_time_series(artist_file_path):
	dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
	data = pd.read_csv(artist_file_path, parse_dates=0, index_col=0, header=None, date_parser=dateparse)
	return data
def get_all_image():
	for i in range(50):
		path = 'data/artist/'+str(i+1)+'.csv'
		data = get_artist_data_as_time_series(path)
		plot_image(i+1,data)
# get_all_image()

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
def test_stationarity(id, timeseries, split):
    
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=split)
    rolstd = pd.rolling_std(timeseries, window=split)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    # plt.show(block=False)
    path = 'image/df/'+str(id)+'.png'
    plt.savefig(path)
    plt.close()
    
    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


def run_check(id):
	# check the time series if it's stable 
	path = 'data/artist/'+str(id)+'.csv'
	split = 21 # use 21 days as a period
	ts = get_artist_data_as_time_series(path)[1]
	moving_avg = pd.rolling_mean(ts,split)
	ts_moving_avg_diff = ts - moving_avg
	ts_moving_avg_diff.dropna(inplace=True)
	test_stationarity(id, ts_moving_avg_diff,split)

def check_time_series():
	for id in xrange(1,51):
		run_check(id)
		print "complete " + str(id) + "*********************"

import statsmodels.api as sm

def run_model(id,param,show_pcf_acf=False,show_predict_result=True,save_predict_result=False):
	# use these show_pcf_acf,show_predict_result,save_predict_result you can control the result for debug
	path = 'data/artist/'+str(id)+'.csv'
	split = 21
	ts = get_artist_data_as_time_series(path)[1] # only get the 'play' column
	ts = pd.Series(ts.values,index=ts.index)
	date_index = pd.date_range('2015-08-30', periods=62, freq='D')
	#  use to show pcf and acf, for 1 diff and 2 diff
	# diff_ts1 = ts.diff(1)
	# diff_ts1.dropna(inplace=True)
	
	# diff_ts2 = ts.diff(2)
	# diff_ts1.dropna(inplace=True)	
	if show_pcf_acf:
		# sm.graphics.tsa.plot_pacf(diff_ts1, lags=40)
		# sm.graphics.tsa.plot_acf(diff_ts1, lags=40)
		sm.graphics.tsa.plot_pacf(ts, lags=40)
		sm.graphics.tsa.plot_acf(ts, lags=40)
		plt.show()
		plt.close()
	
	model = sm.tsa.ARMA(ts, order=param)
	# model = sm.tsa.ARMA(diff_ts1,order=param)
	# results_AR = model.fit(trend='nc', disp=-1)
	results_AR = model.fit()
	# print(results_AR.summary())
	# predict_rs = results_AR.fittedvalues # calculate the origin predict data
	
	if show_predict_result or save_predict_result:
		results_AR.plot_predict(start='2015-03-02',end='2015-10-30')
		# tmp = results_AR.predict(start='2015-08-31',end='2015-10-30')
		# start = ts['2015-08-30']
		# print 'start:',start
		# tmp = pd.Series(tmp,index=date_index)
		# tmp['2015-08-30'] = start
		# tmp = np.cumsum(tmp)
		# print tmp
		if show_predict_result:
			plt.show()
		if save_predict_result:
			path = 'image/predict/'+str(id)+'.png'
			plt.savefig(path)
		plt.close()

run_model(1,(4,2),show_pcf_acf=True)
# run_model(22,(2,1),show_pcf_acf=False)

# for id in xrange(1,51):
# 	# if id in [10,21,22,33,40]: # there has some error in these id, need to be check
# 	# 	continue
# 	if id == 6:
# 		continue
# 	if id == 22 or id == 43:
# 		param = (2,0,0)
# 	elif id == 4 or id ==17 or id == 25:
# 		param = (2,0,2)
# 	else:
# 		param = (2,0,1)
# 	run_model(id,param,show_predict_result=False,save_predict_result=True)
# 	print str(id)+'*********************'