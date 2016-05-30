import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
PNG_DIR	= "image/"
USER_WITH_ARTIST_FILE_PATH = "data/user_with_artist.csv"

# get the numbers of user action for each artist_id
user_with_artist_data = pd.read_csv(USER_WITH_ARTIST_FILE_PATH, index_col=0)
tmp = user_with_artist_data[['artist_id','action_type','Ds']]
tmp['num']=1 # add a column 1 for groupby sum count.
groups = tmp.groupby(['artist_id'])
for aid,group in groups:
	print aid
	print group.groupby(['Ds','action_type']).sum()
	data = group.groupby(['Ds','action_type']).sum().unstack().fillna(0)
	data.plot()
	path = PNG_DIR + aid + ".png"
	plt.savefig(path)

# group = tmp.groupby(['artist_id','action_type','Ds']).sum()





