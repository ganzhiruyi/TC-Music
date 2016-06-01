import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
PNG_DIR	= "image/"
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
get_all_image()
from statsmodels.tsa.arima_model import ARIMA
def run_model(data):
	pass





