import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.externals import joblib

def download_data(URL):
    frame = pd.read_csv(URL, sep='\t', header=None)
    return frame

user_id = []
for i in range(5000, 7000):
    temp = 'u' + str(i)
    user_id.append(temp)

df = DataFrame()
df['uid'] = user_id

sms = download_data('sms_test_a.txt')
sms.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'in_out']

voice = download_data('voice_test_a.txt')
voice.columns = ['uid', 'opp_num', 'opp_head', 'opp_len', 'start_time', 'end_time', 'call_type', 'in_out']
voice['use_time'] = voice['end_time'] - voice['start_time']

wa = download_data('wa_test_a.txt')
wa.columns = ['uid', 'wa_name', 'visit_cnt', 'visit_dura', 'up_flow', 'down_flow', 'wa_type', 'date']

# sms

sms_feat = DataFrame()
gp = sms.groupby('uid')['opp_num']
x0 = gp.apply(lambda x: x.count())
sms_feat['uid'] = x0.index
sms_feat['sms_opp_count_all'] = x0.values

gp = sms.groupby('uid')['opp_num']
x0 = gp.apply(lambda x: len(set(x)))
sms_feat['sms_opp_count_unique'] = x0.values

gp = sms.groupby('uid')['opp_len']
x0 = gp.apply(lambda x: x.unique().mean())
sms_feat['sms_opp_avg_len'] = x0.values

gp = sms.groupby(['uid', 'in_out'])['opp_num']
x0 = gp.apply(lambda x: x.count())
x0 = x0.unstack(fill_value=0).reset_index(drop=True)
x0.columns = ['0', '1']
sms_feat['count_out'] = x0['0']
sms_feat['count_in'] = x0['1']

df = df.merge(sms_feat, on='uid', how='left').reset_index(drop=True)
df = df.fillna(0)

# voice

voice_feat = DataFrame()
gp = voice.groupby('uid')['opp_num']
x0 = gp.apply(lambda x: x.count())
voice_feat['uid'] = x0.index
voice_feat['voice_opp_count_all'] = x0.values

gp = voice.groupby('uid')['opp_num']
x0 = gp.apply(lambda x: len(set(x)))
voice_feat['voice_opp_count_unique'] = x0.values

gp = voice.groupby('uid')['opp_len']
x0 = gp.apply(lambda x: x.unique().mean())
voice_feat['voice_opp_avg_len'] = x0.values

gp = voice.groupby('uid')['use_time']
x0 = gp.apply(lambda  x: x.sum())
voice_feat['voice_total_use_time'] = x0.values

gp = voice.groupby('uid')['use_time']
x0 = gp.apply(lambda  x: x.mean())
voice_feat['voice_avg_use_time'] = x0.values

gp = voice.groupby(['uid', 'in_out'])['opp_num']
x0 = gp.apply(lambda x: x.count())
x0 = x0.unstack(fill_value=0).reset_index(drop=True)
x0.columns = ['0', '1']
voice_feat['count_out'] = x0['0']
voice_feat['count_in'] = x0['1']

gp = voice.groupby(['uid', 'call_type'])['opp_num']
x0 = gp.apply(lambda x: x.count())
x0 = x0.unstack(fill_value=0).reset_index(drop=True)
x0.columns = ['1', '2', '3', '4', '5']
voice_feat['voice_count_type1'] = x0['1']
voice_feat['voice_count_type2'] = x0['2']
voice_feat['voice_count_type3'] = x0['3']
voice_feat['voice_count_type4'] = x0['4']
voice_feat['voice_count_type5'] = x0['5']

df = df.merge(voice_feat, on='uid', how='left').reset_index(drop=True)
df = df.fillna(0)

# wa

wa_feat = DataFrame()
gp = wa.groupby('uid')['visit_cnt']
x0 = gp.apply(lambda x: x.sum())
wa_feat['uid'] = x0.index
wa_feat['wa_visit_cnt_sum'] = x0.values

gp = wa.groupby('uid')['wa_name']
x0 = gp.apply(lambda x: len(set(x)))
wa_feat['wa_name_count_unique'] = x0.values

gp = wa.groupby('uid')['visit_dura']
x0 = gp.apply(lambda x: x.mean())
wa_feat['wa_visit_dura_mean'] = x0.values

gp = wa.groupby('uid')['up_flow']
x0 = gp.apply(lambda x: x.mean())
wa_feat['wa_up_flow_mean'] = x0.values

gp = wa.groupby('uid')['down_flow']
x0 = gp.apply(lambda x: x.mean())
wa_feat['wa_down_flow_mean'] = x0.values

gp = wa.groupby(['uid', 'wa_type'])['wa_name']
x0 = gp.apply(lambda x: x.count())
x0 = x0.unstack(fill_value=0).reset_index(drop=True)
x0.columns = ['0', '1']
wa_feat['wa_count_type0'] = x0['0']
wa_feat['wa_count_type1'] = x0['1']

df = df.merge(wa_feat, on='uid', how='left').reset_index(drop=True)
df = df.fillna(0)

classifier = joblib.load("train_model.m")
test_X = df.drop(['uid'], axis=1)
results = DataFrame()
results['uid'] = df['uid']
results['label'] = classifier.predict(test_X)
results['prob'] = classifier.predict_proba(test_X)[:, 1]
results.reset_index(drop=True)
results.to_csv('results.csv')
