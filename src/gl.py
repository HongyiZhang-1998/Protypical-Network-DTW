import numpy as np

epoch=0
device='cuda:5'
experiment_root='../output'
debug=False
local_match=0
reg_rate=0
threshold=3
gamma=0.1
iter=0
R_=np.random.randn(250, 15, 15)
D_=np.random.randn(250, 15, 15)
mod='train'
run_mode='train'
backbone='st_gcn'
dataset='ntu120'
use_attention=0
use_bias=0
n_class=5
shot=1
n_query=5
classes_iter=[]
dtw=1
eps=1
leave_all_frame=0
num=30