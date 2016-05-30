import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
USER_ACTION_FILE_PATH = "data/mars_tianchi_user_actions.csv"
SONG_FILE_PATH = "data/mars_tianchi_songs.csv"
USER_WITH_ARTIST_FILE_PATH = "data/user_with_artist.csv"

user_index = ['user_id','song_id','gmt_create','action_type','Ds']
song_index = ['song_id','artist_id','publish_time','song_init_plays','Language','Gender']

user_data = pd.read_csv(USER_ACTION_FILE_PATH,	header=None, names=user_index)
song_data = pd.read_csv(SONG_FILE_PATH, header=None, names=song_index)

user_with_artist_data = pd.merge(user_data,song_data[['song_id','artist_id']],how='left',on=['song_id']);
user_with_artist_data.to_csv(USER_WITH_ARTIST_FILE_PATH)

# get the numbers of user action for each artist_id
tmp = user_with_artist_data[['artist_id','Ds']]
tmp['num']=1 # add a column 1 for groupby sum count.
group = tmp.groupby(['artist_id','Ds']).sum()

