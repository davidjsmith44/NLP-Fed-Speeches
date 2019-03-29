'''
This file load up the final data structures for the model
after all preprocessing has happened
'''

import pickle
import os

'''
importing the list of cosine similairities and dates from
the speeches processing (NLP_pipeline file)

the file is saved as a list with three arrays
The first array is the cosine of the speech to the last speech
The second array contains the cosine similairities between the
last speech and the n previous speeches
The third array is the dates corresponding to these speeches

The last 50 speeches are empty because to calculate the average
we needed to start sometime after the first speech. This should be
cleaned up in the NLP_pipeline at a later date, but for now we are
cleaning it here
'''
cosine_list = pickle.load(open('../data/ts_cosine_sim', 'rb'))

# split these into variables
ts_cos_last = cosine_list[0]
ts_cos_avg_n = cosine_list[1]
ts_dates = cosine_list[2]

# cleaning up the emtpy values at the end of the dataframe
# creating an index to the zeros
ind_zeros = ts_cos_last != 0
# cleaning out the three variables
ts_cos_last = ts_cos_last[ind_zeros]
ts_cos_avg_n = ts_cos_avg_n[ind_zeros]
ts_dates = ts_dates[ind_zeros]

