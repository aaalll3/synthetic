# Synthetic

A simple example for synthetic trajectory + mapping of signal values

For usage, after installed requirements:

```bash
python ./exampl.py
```

Functions related to simulate trajectories in sim.py:

```python
# simulation
from sim import astar,markov,map_heur

traj = astar(map_df,x1,y1,x2,y2) # simulation with path finding
traj = markov(map_df,tx,ty) # simulation with markov procecss

```

Functions related to mapping in map.py:

```python
# mapping/transform
from map import Mapper,diffuse

snr_map,wall_df,_,_ = diffuse(pos_data,snr_grid,map_shape,grid_shape) # transofrm with a diffused map
model = Mapper()
model.fit(pos_train,snr_train)
snr_test = model.pred(pos_test) # transform with a model
```

For utility of ploting and loading:

```python
# draw map and trajectory according to its coordinates
draw_value(xp,yp,value=None,title_msg="",marker_size=30,cmap='plasma',marker_shape='.',map_df=map_df,limit=True)

# index to map coordinates casting
itm(i,j,map_df=map_df)

# map coordinates to index casting
mti(x,y,map_df=map_df)

# align column based on keywords in Pandas dataframe
align(df:pd.DataFrame,kwl:list)
```

Data and cooresponding processing code are from https://github.com/fraunhoferhhi/ai4mobile-industrial/tree/main
