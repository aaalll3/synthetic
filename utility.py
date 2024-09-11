import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_path = Path.cwd()/'data'
# iv2i
map_path = data_path/'static_map.parquet'
iv2ip_df = pd.read_parquet(data_path/'iV2Ip.parquet')
map_df = pd.read_parquet(map_path)


def exclude_none(narr,debug=False):
    '''exclude none in arr'''
    return narr != None
def exclude_nan(narr,debug=False):
    '''exclude nan in arr'''
    return ~np.isnan(narr) 
def exclude_all(narr):
    '''exclude none and nan in arr'''
    return exclude_none(narr) & exclude_nan(narr)
def align(df:pd.DataFrame,kwl:list):
    '''exclude none and nan for keywords in kwl
    
    df: target dataframe to extract
    kwl: a list of keywords in df, target attributes
    return valid common index in df 
    '''
    num_entry = (df.shape[0],)# num_entries
    idx_all = np.ones(num_entry,dtype=bool)
    for kw in kwl:
        dim1 = np.array(df[kw])
        idx_v = exclude_all(dim1)
        idx_all = idx_all&idx_v
    return idx_all

def mse(arr1,arr2):
    return ((arr1-arr2)**2).mean()

def plot_map(map_df, ax=None, cmap='Greens'):
    map_cols = np.array(map_df.columns, dtype=np.float32)
    map_rows = np.array(map_df.index, dtype=np.float32)
    extent_df = [map_cols[0], map_cols[-1], map_rows[-1], map_rows[0]]

    if ax is not None:
        plt.sca(ax)
    plt.imshow(map_df, cmap=cmap, extent=extent_df)
    # plt.imshow(map_df, cmap=cmap)

def spatial_avg(df, pos_labels, tile_size, avg_method='mean'):
    x, y = pos_labels

    avg_df = df.copy()
    avg_df[x] = round(avg_df[x] / tile_size) * tile_size
    avg_df[y] = round(avg_df[y] / tile_size) * tile_size
    avg_df = avg_df.groupby([x, y]).aggregate(avg_method, numeric_only=True).reset_index()

    return avg_df

def draw_value(xp,yp,value=None,title_msg="",marker_size=30,cmap='plasma',marker_shape='.',map_df=map_df,limit=True):
    '''draw a path on map, optional provide value of the path
    
    xp,yp: x,y coordinate in 1-d array
    value: any value cooresponding to x,y in 1-d array, must be same shape to xp/yp
        if value=None, path will paint in red
    title_msg: customized name of the plot
    '''
    bs_x = 9
    bs_y = 9
    
    pos_x = 'position_x'
    pos_y = 'position_y'
    df_grid = spatial_avg(iv2ip_df, pos_labels=(pos_x, pos_y), tile_size=0.1)
    df_grid['obstacles_log'] = np.log10(df_grid['obstacles_sum']+1)
    # df_grid.plot(pos_x, pos_y, kind='scatter', c='serving_cell_snr_1', s=5, cmap="plasma", ax=ax, ylim=(-25,30))
    # df_grid.plot(pos_x, pos_y, kind='scatter', c=snr, s=5, cmap="plasma", ax=ax, ylim=(-25,30))

    plt.figure(figsize=(10,6))
    plot_map(map_df, cmap='Greens')
    # plot_map_np(map_cols,map_rows,map_data,cmap='Greens')
    ax = plt.gca()
    if value is not None:
        if limit:
            sc = ax.scatter(xp,yp,c=value,cmap=cmap,s=marker_size,marker=marker_shape,vmin=-6,vmax=25)
        else:
            sc = ax.scatter(xp,yp,c=value,cmap=cmap,s=marker_size,marker=marker_shape)
    else:
        sc = ax.scatter(xp,yp,c='red',s=marker_size,marker=marker_shape)
    # sc = ax.scatter(xp[-1],yp[-1],c=value[-1],cmap='plasma',s=marker_size,marker='.')

    plt.scatter(bs_x, bs_y, marker='^', c='red', s=100)
    _ = plt.text(bs_x, bs_y, 'Base\nstation', horizontalalignment='center', verticalalignment="bottom")
    _ = plt.title(f"{title_msg} iV2I+ visualization - Heatmap of the cleaning AGV")
    plt.colorbar(sc)
    plt.ion()
    # plt.show()

def draw_value_hit(xp,yp,value,x2,y2,v2,title_msg="",marker_size=30,cmap='plasma',cmap2='rainbow',marker_shape='.',map_df=map_df):
    '''draw a path on map, optional provide value of the path
        with red cross in the plot 
    
    xp,yp: path's x,y coordinate in 1-d array
    value: any value cooresponding to x,y in 1-d array, must be same shape to xp/yp
        if value=None, path will paint in red
        
    x2,y2: cross' x,y coordinates in 1-d array
    value: any value cooresponding to x,y in 1-d array, must be same shape to x2/y2
        if value=None, path will paint in red
    title_msg: customized name of the plot
    '''
    bs_x = 9
    bs_y = 9
    
    pos_x = 'position_x'
    pos_y = 'position_y'
    df_grid = spatial_avg(iv2ip_df, pos_labels=(pos_x, pos_y), tile_size=0.1)
    df_grid['obstacles_log'] = np.log10(df_grid['obstacles_sum']+1)

    plt.figure(figsize=(10,6))
    plot_map(map_df, cmap='Greens')
    # plot_map_np(map_cols,map_rows,map_data,cmap='Greens')
    ax = plt.gca()
    if value is not None:
        sc = ax.scatter(xp,yp,c=value,cmap=cmap,s=marker_size,marker=marker_shape)
    else:
        sc = ax.scatter(xp,yp,c='red',s=marker_size,marker=marker_shape)
    # sc = ax.scatter(xp[-1],yp[-1],c=value[-1],cmap='plasma',s=marker_size,marker='.')

    if v2 is not None:
        sc = ax.scatter(x2,y2,c=v2,s=marker_size+30,marker='x')
    else:
        sc = ax.scatter(x2,y2,c='red',s=marker_size+30,marker='x')
    
    plt.scatter(bs_x, bs_y, marker='^', c='red', s=100)
    _ = plt.text(bs_x, bs_y, 'Base\nstation', horizontalalignment='center', verticalalignment="bottom")
    _ = plt.title(f"{title_msg} iV2I+ visualization - Heatmap of the cleaning AGV")
    plt.colorbar(sc)
    plt.ion()
    # plt.show()
# utility
def itm(i,j,map_df=map_df):
    '''i,j to x,y full map
    '''
    map_cols = np.array(map_df.columns, dtype=np.float32)
    map_rows = np.array(map_df.index, dtype=np.float32)
    return map_cols[i],map_rows[j]
    
def mti(x,y,map_df=map_df):
    '''x,y to i,j full map'''
    map_cols = np.array(map_df.columns, dtype=np.float32)
    map_rows = np.array(map_df.index, dtype=np.float32)
    x_num = map_cols.shape[0]
    y_num = map_rows.shape[0]
    xs = map_cols[0]
    xe = map_cols[-1]
    ys = map_rows[0]
    ye = map_rows[-1]
    i = ((x-xs)*x_num/(xe-xs)).astype(np.int32)
    j = ((y-ys)*y_num/(ye-ys)).astype(np.int32)
    # print(map_rows)
    # print(y)
    # print(j)
    return i,j

def aitm(i,j,shape,bound):
    '''arbitrary ij to map coord, with crop and resize
        ij,array or scalar
        shape [rows,cols]
        bound [xmin,ymin,xmax,ymax] left-bottom -> right-top
        xmin,ymin,xmax,ymax=0,0,477,864
        output -> x,y coord
    '''
    x_num = shape[1]
    y_num = shape[0]
    xs,ys,xe,ye = bound
    x = (xe-xs)/x_num*i+xs
    y = (ye-ys)/y_num*j+ys
    return x,y
    
def amti(x,y,shape,bound):
    '''arbitrary map coord to ij, with crop and resize
        xy,array or scalar
        shape [rows,cols]
        bound [xmin,ymin,xmax,ymax] left-bottom -> right-top
        xmin,ymin,xmax,ymax=0,0,477,864
        output -> i,j idx
    '''
    x_num = shape[1]
    y_num = shape[0]
    xs,ys,xe,ye = bound
    i = ((x-xs)*x_num/(xe-xs)).astype(np.int32)
    j = ((y-ys)*y_num/(ye-ys)).astype(np.int32)
    return i,j

def gridy(xmin,xmax,ymin,ymax,stepx=0.7,stepy=0.7):
    '''form a grid base on x,y limitation and step length between
    '''
    x_segment = np.arange(xmin,xmax,stepx)
    y_segment = np.arange(ymin,ymax,stepy)
    reps_x = y_segment.shape[0]
    reps_y = x_segment.shape[0]
    num_rows=reps_x
    num_cols=reps_y
    x_segment = np.tile(x_segment,reps=reps_x) # e.g.123123
    y_segment = np.repeat(y_segment,repeats=reps_y) # e.g.112233
    z_segment = np.ones(x_segment.shape)
    # z_grid = z_grid*pos_data.mean(axis=0)[2]
    pos_grid = np.vstack((x_segment,y_segment,z_segment)).T
    print(f'grid shape check {reps_x}*{reps_y}={reps_x*reps_y} x:{x_segment.shape} y:{y_segment.shape} grid:{pos_grid.shape}')
    return pos_grid,num_rows,num_cols

def define_wall_df(map_df,th=35,wall=20000,void=50000):
    '''define wall value
    
    th: threshold of identify a wall, greater is wall, less is empty
    wall: assigned value for wall
    void: assigned value for void place
    '''
    map_cols = np.array(map_df.columns, dtype=np.float32)
    map_rows = np.array(map_df.index, dtype=np.float32)
    map_data = map_df.values
    nan_data_idx = np.isnan(map_data)
    map_data[nan_data_idx] = void
    gt = map_data >th
    map_data[gt]=wall
    map_data[~gt]=0
    map_data = np.float32(map_data)
    wall_df = pd.DataFrame(data=map_data,index=map_rows,columns=map_cols)
    return wall_df
