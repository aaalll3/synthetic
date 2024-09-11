import os
import math
from queue import PriorityQueue

import numpy as np

from utility import exclude_none, exclude_nan, exclude_all, align
from utility import mse
from utility import itm,mti,aitm,amti
from utility import gridy
from utility import define_wall_df
from utility import draw_value, draw_value_hit

def heur(xs,ys,xe,ye):
    '''simplel heuristic function towards endpoint
    
    xs,ys: start point coord
    xe,ye: end point coord
    '''
    # return np.sqrt((xs-xe)**2+(ys-ye)**2)
    return np.abs(xs-xe)*10+np.abs(ys-ye)*10

def map_heur(map_df,wall_th=20000,void_th=50000): 
    '''compute a heuristical map for encouraging central path
    more far away from wall, lower the value
    close to way higher the value
    
    wall_th: penalty value of wall in map
    void_th: penalty value of void space in map
    return a map of penalty value represents distance to wall
    '''
    # map access with (j,i) order
    map_cols = np.array(map_df.columns, dtype=np.float32)
    map_rows = np.array(map_df.index, dtype=np.float32)
    map_shape=map_df.shape
    map_data= map_df.values
    distance_map = np.zeros(map_data.shape)
    # wall list
    void_idx=map_data >= void_th
    invalid_idx=map_data >= wall_th
    wall_idx=(invalid_idx.astype(np.int32) - void_idx.astype(np.int32)).astype(bool)
    space_idx=~invalid_idx
    
    def mat2idx(idxs,m,n):
        # mat m row*n cols to idx, thus with (j,i) order
        num = np.arange(0,idxs.size,1,np.int32)
        num = num[idxs.flatten()]
        y,x = np.divmod(num,n)
        coord = np.vstack([y,x]).T
        coord = list(map(tuple,coord))
        return coord
    
    def mat2idx_tp(idx,shape):
        return mat2idx(idx,shape[0],shape[1])
    
    def checkBound(xylist):
        xmin,ymin,xmax,ymax=0,0,477,864
        # xmin=0
        # xmax=477
        # ymin=0
        # ymax=864
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
    def getNeighbor(x,y,bound=None):
        # bound(left-bottom to right-top(exclude))
        if bound is None:
            xmin,ymin,xmax,ymax=0,0,477,864
        else:
            xmin,ymin,xmax,ymax=bound[0],bound[1],bound[2],bound[3]
        d=1
        pend=np.arange(-d,d+1,dtype=np.int32)
        xn = pend+x
        yn = pend+y
        reps = len(yn)
        xn = np.tile(xn,reps)
        yn = np.repeat(yn,reps)
        nn = np.vstack([yn,xn]).T
        valids = list(map(checkBound,nn))
        nn = nn[valids]
        nn = list(map(tuple, nn))
        return nn
    
    def forward(front,front_level,distance_map,observed):
        # front: [(j,i)...]
        # front_step: current front's steps away from wall
        # distance_map: store distancec to wall
        # observed: bool, map shape, (j,i) order
        newfront = []
        for cur in front:
            nn = getNeighbor(cur[1],cur[0])
            for n in nn:
                if not observed[n]:
                    observed[n]=True
                    distance_map[n] = front_level+1
                    newfront.append(n)
        return newfront,distance_map,observed
            
    front = mat2idx_tp(wall_idx,map_shape)
    observed=invalid_idx.copy()
    front_level=0
    while len(front)!=0:
        front,distance_map,observed = forward(front,front_level,distance_map,observed)
        front_level += 1
    return distance_map

class Fitem(object):
    '''define item in priority queue, used in A* algorithm
    
    priority: a score for sorting
    x,y: coord of this element
    self.neighbor: return neighbor of x,y with in a range d
    self.loc: return coord in yx order
    '''
    def __init__(self,cost,x,y):
        self.priority = cost
        self.x = int(x)
        self.y = int(y)
 
    def __str__(self):
        return f"Fitem(priority={self.priority}, x,y:({self.x},{self.y}))"
 
    def __lt__(self, other):
        return self.priority < other.priority
    
    def checkBound(self,x,y):
        '''chech if x,y coord out of bound'''
        xmin,ymin,xmax,ymax=0,0,477,864
        if x >= xmax:
            return False
        if x < xmin:
            return False
        if y >= ymax:
            return False
        if y < ymin:
            return False
        return True
    
    def neighbor(self):
        ''' return a list of neighbor of self.x,self.y within a range of d=3
        
        return with [(y1,x1),(y2,x2) ... ]'''    
        def checkBound(xylist):
            xmin,ymin,xmax,ymax=0,0,477,864
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
        
        x=self.x
        y=self.y
        d=3
        pend=np.arange(-d,d+1,dtype=np.int32) #2d+1
        xn = pend+x
        yn = pend+y
        reps = len(yn)
        xn = np.tile(xn,reps)
        yn = np.repeat(yn,reps)
        nn = np.vstack([yn,xn]).T
        valids = list(map(checkBound,nn))
        nn = nn[valids]
        nn = list(map(tuple, nn))
        return nn
    
    def loc(self):
        return (self.y,self.x)

def astar(map_df,xs,ys,xe,ye,th=1000000,central=None):
    '''A* algorithm for find shortest path with a costumized heuristic functioin
    
    map_df: map in dataframe
    xs,ys: start coord
    xe,ye: end coord
    th: threshold of max iteration number
    central: heurstic map for encouraging go through central of the space
    '''
    print(f'astar')
    map_shape=map_df.shape
    map_data= map_df.values
    step_penalty = 1
    ctc = 10
    # print(map_shape)
    # return
    fscore = PriorityQueue()
    if central is not None:
        fscore.put(Fitem(heur(xs,ys,xe,ye) + central[(ys,xs)]*ctc ,xs,ys))
    else:
        fscore.put(Fitem(heur(xs,ys,xe,ye),xs,ys))
    gscore=np.ones(map_shape)
    gscore = gscore*1000000
    gscore[ys,xs]=0
    cand=[]
    path=[]
    if central is not None:
        central_reward =  central
    pathx=np.zeros(map_shape,dtype=np.int32)
    pathy=np.zeros(map_shape,dtype=np.int32)
    cnt=0
    while fscore.empty() != True:
        # print(f'cnt:{cnt} queue:{fscore.empty()}')
        cur = fscore.get(timeout=3)
        # print(f'get')
        cur_loc = cur.loc()
        cur_g = gscore[cur_loc]
        # print(f'#{cnt} at{cur.loc()}')
        cnt+=1
        vec_o =  np.array(cur_loc)
        if cnt > th:
            print('out of steps')
            print(pathy.shape)
            cur_loc = cur.loc()
            while cur_loc[0]!=ys or cur_loc[1]!=xs: 
                path.append(cur_loc)
                cur_loc=(pathy[cur_loc],pathx[cur_loc])
            return path
        if cur_loc[0]==ye and cur_loc[1]==xe:
            print(f'found a path from ({xs},{ys}) to ({ys},{ye})')
            print(pathy.shape)
            cur_loc = cur.loc()
            while cur_loc[0]!=ys or cur_loc[1]!=xs: 
                path.append(cur_loc)
                cur_loc=(pathy[cur_loc],pathx[cur_loc])
            return path
        
        for n in cur.neighbor():
            # print(f'touch {n}-{type(n)}')
            if n == cur_loc:
                continue
            vec_n = np.array(n)
            length_penalty = np.sqrt(np.linalg.norm(vec_n-vec_o,2))
            if central is not None:
                score = cur_g + map_data[n] + length_penalty + step_penalty + central[n]*ctc

            else:
                score = cur_g + map_data[n] + length_penalty + step_penalty
            # score = cur_g + map_data[n] + step_penalty
            # print(f'my neighbor {cur_loc} {n}')
            if score < gscore[n]:
                # print('push')
                pathy[n]=cur_loc[0]
                pathx[n]=cur_loc[1]
                gscore[n]=score
                if central is not None:
                    # print(f'push {n[1]},{n[0]}@{heur(n[1],n[0],xe,ye)+score + central[n[1],n[0]]*ctc}')
                    item = Fitem(heur(n[1],n[0],xe,ye)+score + central[n]*ctc,n[1],n[0])
                else:
                    item = Fitem(heur(n[1],n[0],xe,ye)+score,n[1],n[0])
                fscore.put(item)

def markov(map_df,xs,ys,stride=0.5,steps=200):
    '''generate a trajectory based on Markov random process
    
    xs,ys: start point
    stride: path length of each random walk
    steps: number of steps in the process
    '''
    wall_df = define_wall_df(map_df)
    wall_data = wall_df.values
    map_shape=map_df.shape
    map_data= map_df.values
    passed = np.zeros(wall_data.shape,dtype=bool)
    pos_x=[xs]
    pos_y=[ys]
    
    xc=xs
    yc=ys
    while len(pos_x)<steps:
        angle = np.random.uniform(0,2*np.pi)
        walk_x = np.cos(angle)*stride
        walk_y = np.sin(angle)*stride
        next_x = xc+walk_x
        next_y = yc+walk_y
        n_i,n_j= mti(next_x,next_y,map_df)
        if wall_data[n_j,n_i]>1000:
            continue
        if passed[n_j,n_i]:
            continue
        passed[n_j,n_i]=True
        xc = next_x
        yc = next_y
        pos_x.append(next_x)
        pos_y.append(next_y)
        
    return pos_x,pos_y
