import os
import math
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.neighbors import KNeighborsRegressor

from utility import exclude_none, exclude_nan, exclude_all, align
from utility import mse
from utility import itm,mti,aitm,amti
from utility import gridy
from utility import define_wall_df
from utility import draw_value, draw_value_hit

from sim import astar,markov,map_heur

from map import Mapper,diffuse

###------------------------keywords in dataset---------------------------###
# v2i
# basic
itime = "time[s]"   
# signal
rsrp = "serving_cell_rsrp_1"
rsrq = "serving_cell_rsrq_1"
rssi = "serving_cell_rssi_1"
snr = "serving_cell_snr_1"
# net
ping = "ping_ms"
js = "jitter_server"
jc = "jitter_client"
thu = "throughput_UL"
thd = "throughput_DL"
taru = "target_UL"
tard = "target_DL"

ttl = "ttl"
icmp_seq = "icmp_seq"
#physical
xp = 'position_x'
yp = 'position_y'
zp = 'position_z'
ox = 'orientation_x'
oy = 'orientation_y'
oz = 'orientation_z'
ow = 'orientation_w'
vx = 'twist_linear_x'
vy = 'twist_linear_y'
vz = 'twist_linear_z'
ax = 'twist_angular_x'
ay = 'twist_angular_y'
az = 'twist_angular_z'
bs = 'distance_to_bs'
los = 'line_of_sight'

###---------------------------extract data from dataframe-----------------------------###

data_path = Path.cwd()/'data'
# iv2i
map_path = data_path/'static_map.parquet'
iv2ip_df = pd.read_parquet(data_path/'iV2Ip.parquet')
map_df = pd.read_parquet(map_path)
## map conf   
map_cols = np.array(map_df.columns, dtype=np.float32)
map_rows = np.array(map_df.index, dtype=np.float32)
xmin = np.min(map_cols)
xmax = np.max(map_cols)
ymin = np.min(map_rows)
ymax = np.max(map_rows)
print(f'xmin:{xmin},xmax:{xmax},ymin:{ymin},ymax:{ymax}')

idx_boundary=[0,0,map_rows.shape[0],map_cols.shape[0]] # exclude
map_shape=[map_rows.shape[0],map_cols.shape[0]] # exclude
map_crop_bound=[xmin,ymin,xmax,ymax]

# add keyword
kw_pos_snr = [snr,xp,yp,zp]
pos_snr_idx = align(iv2ip_df,kw_pos_snr)
print(pos_snr_idx)

###---------------------------prepare data for train and test-----------------------------###
pos_snr_data = np.array(iv2ip_df[kw_pos_snr])
pos_snr_data = pos_snr_data[pos_snr_idx]
print(f'extract shape:{pos_snr_data.shape}')

pos_data = pos_snr_data[:,1:]
snr_data = pos_snr_data[:,0]
print(pos_data.shape)


## select train data
train_start = 0
train_end = 300
pos_xyz_train=pos_data[train_start:train_end,:]
snr_train=snr_data[train_start:train_end]

## select test data
print(f'predict')
num_pos = pos_data.shape[0]
a = random.randint(train_end,num_pos)
b = random.randint(train_end,num_pos)
pred_start = None
pred_end = None
if a>b:
    pred_start = b
    pred_end = a  
else:
    pred_start = a
    pred_end = b
show_start = pred_start
show_end = pred_end

color=np.ones(show_end-show_start)

pos_xyz_test = pos_data[pred_start:pred_end,:]
snr_test = snr_data[pred_start:pred_end]

## train machine learning map: svr

model = Mapper()
model.fit(pos_xyz_train,snr_train)
model.pred(pos_xyz_test)

## reconstruct
snr_recon = model.pred(pos_xyz_train)
# compare error to ground truth
print(f'@train interval {train_start}-{train_end} MSE:{mse(snr_recon,snr_train)}')


## prepare diffuse method

## form a grid map
wall_df = define_wall_df(map_df)

step = 0.5 # parameter for grid granularity
stepx = step
stepy = step
pos_grid,rows_grid,cols_grid = gridy(xmin,xmax,ymin,ymax,stepx,stepy)
grid_shape = [rows_grid,cols_grid ]
snr_grid = model.pred(pos_grid)

## train diffuse map
snr_new,wall_new_df,cands,cands_v = diffuse(pos_data=pos_data,snr_grid=snr_grid,map_shape=map_shape,grid_shape=grid_shape,crop_bound=map_crop_bound,wall_df=wall_df)
snr_new=snr_new.flatten()

knn = KNeighborsRegressor()
knn.fit(pos_grid[:,0:2],snr_new)

show_diffused = False
if show_diffused:
    draw_value(pos_grid[:,0],pos_grid[:,1],snr_new,title_msg='diffuse',marker_size=13,limit=True)

map_df = define_wall_df(map_df)

# example point
bs_x = 9
bs_y = 9

mxs,mys=1.7,24.7
mx1,my1=-3.6,5.6
mxe,mye=-3.9,-10

ms_xs,ms_ys=-16.5,1.5
ms_xe,ms_ye=-4.8, 5.1

tx,ty = -0.9,15.8

##----------------simulation shortest path--------------##
dist_map = map_heur(map_df)
dist_df = pd.DataFrame(data=dist_map, index=map_rows,columns=map_cols)
central_reward = dist_map
central_reward = np.max(central_reward) - central_reward

## control point        
show_control_point = False
if show_control_point:
    mi,mj = mti([mxs,mx1,mxe],[mys,my1,mye])

    path1 = astar(map_df,mi[1],mj[1],mi[0],mj[0],central=central_reward)

    path2 = astar(map_df,mi[2],mj[2],mi[1],mj[1],central=central_reward)
    print(f'astar path {len(path1)} + {len(path2)}')

    path1 = np.array(path1,dtype=np.int32)
    path2 = np.array(path2,dtype=np.int32)
    py1 = path1[:,0]
    px1 = path1[:,1]
    py2 = path2[:,0]
    px2 = path2[:,1] 
    mx = np.concatenate([px2,px1])
    my = np.concatenate([py2,py1])
    mx,my=itm(mx,my)
    mz = np.arange(stop = mx.shape[0],step=1,dtype=np.int32)
    cpx = [mxs,mx1,mxe]
    cpy = [mys,my1,mye]
    draw_value_hit(mx,my,mz,cpx,cpy,'red',title_msg='A* trajectory',cmap='brg',marker_size=15) # draw A* path 
    
    ##----------------transform diffuse--------------##
    
    path = np.vstack([mx,my]).T
    path_data = knn.predict(path)
    draw_value(path[:,0],path[:,1],path_data,title_msg='transformed A* trajectory') 

## single run
show_single_run = True
if show_single_run:
    mi,mj = mti([ms_xs,ms_xe],[ms_ys,ms_ye])

    path1 = astar(map_df,mi[1],mj[1],mi[0],mj[0],central=central_reward)
    print(f'astar path len:{len(path1)}')

    path1 = np.array(path1,dtype=np.int32)
    py1 = path1[:,0]
    px1 = path1[:,1]
    mx,my=itm(px1,py1)
    mz = np.arange(stop = mx.shape[0],step=1,dtype=np.int32)
    draw_value(mx,my,mz,title_msg='A* trajectory',cmap='brg',marker_size=15,limit=False) # draw A* path
    
    ##----------------transform diffuse--------------##

    path = np.vstack([mx,my]).T
    path_data = knn.predict(path)
    draw_value(path[:,0],path[:,1],path_data,title_msg='transformed A* trajectory') 

print()
plt.pause(0.1)
input('press Enter to continue')


##----------------simulation markov--------------##
pos_x,pos_y=markov(map_df,tx,ty,stride=0.7,steps=3000)
pos_x=np.array(pos_x)
pos_y=np.array(pos_y)
pos_z = np.arange(stop = pos_x.shape[0],step=1,dtype=np.int32)
draw_value(pos_x,pos_y,pos_z,title_msg='markov trajectory',cmap='brg',marker_size=15,limit=False) # draw path

##----------------transform markov--------------##

path = np.vstack([pos_x,pos_y]).T
path_data = knn.predict(path)
draw_value(path[:,0],path[:,1],path_data,title_msg='transformed markov trajectory') # draw map


plt.pause(0.1)
input('press Enter to continue')



