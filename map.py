import os
import math

import numpy as np
import pandas as pd

from sklearn import svm
from pathlib import Path
from skimage.transform import resize

from utility import exclude_none, exclude_nan, exclude_all, align
from utility import mse
from utility import itm,mti,aitm,amti
from utility import gridy
from utility import define_wall_df
from utility import draw_value, draw_value_hit

'''usage direction
diffuse function and Mapper.pred function can be called separately
for mapping synthetic trajectories to signal values

other functions are helping function for diffuse function

trustful is a customizable function returns a list of coordinates 
which are ground truth 

checkBound,arrBound are functions check if an index out of range 
of the map

getNeighbor,arrNeighbor are functions returns neighbor within a 
distance to given coordinates

align_wall is a function to resize the map(and wall) 
into any given shape
'''

# loading map data
data_path = Path.cwd()/'data'
map_path = data_path/'static_map.parquet'
iv2ip_df = pd.read_parquet(data_path/'iV2Ip.parquet')
map_df = pd.read_parquet(map_path)

## diffuse

def align_wall(wall_df,new_shape,true_coord_bound):
    ''' align wall with grid shape, with possible crop or resize
    
        wall_df: origin value of points in map shape
        new_shape: reshape to this resolution
        crop_bound: crop subset base on coord map [left-bottom,right-top]
        output wall in arbitrary resolution
    '''
    xs,ys,xe,ye = true_coord_bound
    wall_data = wall_df.values
    org_shape = wall_data.shape
    # crop to bound
    i_s,j_s=amti(xs,ys,shape=org_shape,bound=true_coord_bound)
    i_e,j_e=amti(xe,ye,shape=org_shape,bound=true_coord_bound)

    wall_data = wall_data[j_s:j_e,i_s:i_e]
    # resize and normalize
    wmax = np.max(wall_data)
    wall_data = wall_data.astype(np.float32)
    wall_data = wall_data/wmax
    wall_new = resize(wall_data, new_shape, anti_aliasing=False)
    wall_new = wall_new*wmax
    wall_new = wall_new.astype(np.int32)
    rows = np.arange(true_coord_bound[3],true_coord_bound[1],(true_coord_bound[1]-true_coord_bound[3])/new_shape[0],dtype=np.float32)
    cols = np.arange(true_coord_bound[0],true_coord_bound[2],(true_coord_bound[2]-true_coord_bound[0])/new_shape[1],dtype=np.float32)
    coord_bound=true_coord_bound
    idx_bound=[0,0,new_shape[1],new_shape[0]]
    wall_new_df = pd.DataFrame(data=wall_new,index=rows,columns=cols)
    wall_new = wall_new[::-1,:]
    return wall_new,wall_new_df,coord_bound,idx_bound

def trustful(pos_data,grid_shape,crop_bound):
    '''
    return trustful coordinate provided by dataset
    output->ij order
    '''
    xx = pos_data[:,0]
    yy = pos_data[:,1]
    ii,jj = amti(xx,yy,grid_shape,crop_bound)
    trust=np.vstack([jj,ii],dtype=np.int32).T
    ob = np.zeros(grid_shape,dtype=bool)
    single_=[]
    for nn in trust:
        if ~ob[nn[0],nn[1]]:
            ob[nn[0],nn[1]]=True
            single_.append(nn)
    trust = np.array(single_)
    trust=list(map(tuple,trust))
    return trust

def checkBound(xylist,idx_bound):
    '''check if point out of range of idx_bound
    
    idx_bound =[left bottom, right top]
    '''
    xmin,ymin,xmax,ymax=idx_bound
    y = xylist[0]
    x = xylist[1]
    if x >= xmax:
        return False
    if x < xmin:
        return False
    if y >= ymax:
        return False
    if y < ymin:
        return False
    return True

def arrBound(xy_arr,idx_bound):
    '''check if point out of range of idx_bound
    return numpy bool array
    
    xy_arr [[x1,y1],[x2,y2]] xy order
    idx_bound =[left bottom, right top] exclude
    '''
    xmin,ymin,xmax,ymax=idx_bound
    x = xy_arr[:,0]
    y = xy_arr[:,1]
    x_under = x < xmax
    x_up = x>=xmin
    y_under = y < ymax
    y_up = y>=ymin
    return x_under & x_up & y_under & y_up

def getNeighbor(x,y,idx_bound,myself=False):
    '''return list of all neighbors of coords given in x,y
    
        x,y: coord of quering point
        idx_bound(left-bottom to right-top(exclude))
        output -> y,x order
    ''' 
    if idx_bound is None:
        xmin,ymin,xmax,ymax=0,0,477,864
    else:
        xmin,ymin,xmax,ymax=idx_bound[0],idx_bound[1],idx_bound[2],idx_bound[3]
    d=1
    pend=np.arange(-d,d+1,dtype=np.int32)
    xn = pend+x
    yn = pend+y
    reps = len(yn)
    xn = np.tile(xn,reps)
    yn = np.repeat(yn,reps)
    if ~myself:
        ruleoutme=np.ones(yn.shape,dtype=bool)
        ruleoutme[int((2*d+1)**2/2)]=False
        xn=xn[ruleoutme]
        yn=yn[ruleoutme]
    nn = np.vstack([xn,yn]).T
    valids = arrBound(nn,idx_bound)
    nn = nn[valids]
    nn = nn[:,::-1]
    nn = list(map(tuple, nn))
    return nn

def arrNeighbor(x_arr,y_arr,idx_bound):
    '''return list of all neighbors of coords given in x_arr,y_arr
    
        x_arr,y_arr: coords of quering point
        idx_bound: (left-bottom to right-top(exclude))
        output-> yx order
    '''
    if idx_bound is None:
        xmin,ymin,xmax,ymax=0,0,477,864
    else:
        xmin,ymin,xmax,ymax=idx_bound[0],idx_bound[1],idx_bound[2],idx_bound[3]
    d=1
    el=2*d+1
    pend=np.arange(-d,d+1,dtype=np.int32)
    pend_x = pend
    pend_y = pend
    reps = len(pend_y)
    pend_x = np.tile(pend_x,reps)
    pend_y = np.repeat(pend_y,reps)  # (2d+1)^2
    
    print(f'x pend:{pend_x.shape} {pend_x}')
    print(f'y pend:{pend_y.shape} {pend_y}')
    
    x_mat = np.tile(x_arr,el**2).reshape((el**2,x_arr.shape[0])).T
    y_mat = np.tile(y_arr,el**2).reshape((el**2,y_arr.shape[0])).T # num X (2d+1)^2
    print(f'x_mat:{x_mat.shape} {x_mat[0]} {x_mat[1]}')
    print(f'x_mat:{y_mat.shape} {y_mat[0]} {y_mat[1]}')
    print(f'x_mat+pend:{np.add(x_mat,pend_x).shape} {np.add(x_mat,pend_x)[0]} {np.add(x_mat,pend_x)[1]}')
    xn = np.add(x_mat,pend_x).flatten() # broadcast to ax 2
    yn = np.add(y_mat,pend_y).flatten()
    nn = np.vstack([xn,yn]).T # num*(2d+1)^2 X 2
    valids = arrBound(nn,idx_bound)
    nn = nn[valids]
    nn = nn[:,[1,0]] # to yx order
    print(f'nn {nn.shape}')
    nn = list(map(tuple, nn))
    return nn

def diffuse(pos_data,snr_grid,map_shape,grid_shape,crop_bound,wall_df):
    '''Diffuse value to unsupported location
    Given a initial trustful coords, for each iteration, 
    diffuse one step further to neighbor of current trustful coords
    
    pos_data: position data samples in trainset
    snr_grid: all snr data
    map_shape: same as snr_grid in (rows X cols)
    grid_shape: new shape of output
    crop_bound: avaliable bound base on map coord
    wall_df: find wall pos
    '''
    
    map_bs_x=9
    map_bs_y=9
    bs_i,bs_j =amti(map_bs_x,map_bs_y,map_shape,crop_bound)
    th =10000
    wall_data = wall_df.values
    wall_map,wall_new_df,new_coord_b,new_idx_b = align_wall(wall_df,grid_shape,crop_bound)
    
    wall_value_v = wall_map.flatten()
    wall_idx_v = wall_value_v>th # found walls
    
    trust = trustful(pos_data,grid_shape,crop_bound) # y-x order
    
    observed =np.zeros(grid_shape,dtype=bool)
    hitwalls=np.zeros(grid_shape,dtype=bool)
    snr_new =np.zeros(grid_shape,dtype=np.float32)
    
    # init snr wall
    snr_new = snr_new.flatten()
    snr_new[wall_idx_v]=-7
    snr_new = snr_new.reshape(grid_shape)
    # init ob wall
    observed = observed.flatten()
    hitwalls = hitwalls.flatten()
    observed[wall_idx_v]=True
    hitwalls[wall_idx_v]=True
    observed = observed.reshape(grid_shape)
    hitwalls = hitwalls.reshape(grid_shape)
    # init ob trust
    # init snr trust
    print(f'{snr_grid.shape} -- {grid_shape}')
    snr_grid = snr_grid.reshape(grid_shape)
    for sup in trust:
        observed[sup]=True
        snr_new[sup]=snr_grid[sup]

    trust=np.array(trust)
    front = arrNeighbor(trust[:,1],trust[:,0],new_idx_b)
    new_front=[]
    
    def check_ob(yx):
        '''check if yx coord is observed'''
        return observed[yx]
    
    def check_wl(yx):
        '''check if yx coord is wall'''
        return hitwalls[yx]
    
    def diff(yx):
        '''compute current position(yx) value according to neighbor
        
            take global observed new_front snr_new snr_data
        '''
        if ~observed[yx]:
            observed[yx]=True
            nn = getNeighbor(yx[1],yx[0],new_idx_b)
            nn_arr=np.array(nn)
            valid = np.array(list(map(check_ob,nn)))
            hit = np.array(list(map(check_wl,nn)))
            cand = nn_arr[valid&~hit]
            print(f'yx:{yx}---cand:{cand}')
            unob = list(map(tuple,nn_arr[~valid]))
            
            n_cand = []
            for i in range(len(nn)):
                if observed[nn[i]] and not hitwalls[nn[i]]:
                    n_cand.append(nn[i])
            n_cand_v = []
            for one in n_cand:
                n_cand_v.append(snr_new[one])
            n_cand_v = np.array(n_cand_v)
            #para
            take = snr_new[cand].flatten()
            snr_new[yx]=n_cand_v.mean()*np.power(np.e,-cnt/160) #TODO
            new_front.extend(unob)
        
    cnt =0
    cands=[]
    cands_v=[]
    while len(front) != 0:
        cnt+=1
        print(f'depth:{cnt} len:{len(front)}')
        for px in front:
            diff(px)
        cands.append(np.array(front))
        cand_v = []
        for px in front:
            cand_v.append(snr_new[px])
        cands_v.append(np.array(cand_v))
        front=new_front
        new_front=[]
        
    return snr_new,wall_new_df,cands,cands_v

class Mapper(object):
    '''a simple svr model doing regression task from x,y coord to signal value'''
    def __init__(self):
        self.svr_rbf = svm.SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        return
    
    def fit(self,x,y):
        self.svr_rbf.fit(x,y)
        return
    
    def pred(self,x):
        pred = self.svr_rbf.predict(x)
        return pred
