import json
import plotly
import numpy as np
import torch.nn as nn
from torch import FloatTensor
from torch import load
from torch import from_numpy
from scipy.special import fresnel
from scipy.stats import norm

from flask import Flask
from flask import render_template, request 
from plotly.graph_objs import Scatter
from plotly import tools


app = Flask(__name__)


# make a class/def for this
# create the basis set for the simulation and fitting modules
r = np.arange(10,100.1,0.1)
t = np.arange(-0.2,2.8,0.01)*1e-6
len_r = len(r)
len_t = len(t)
w = 326.977e+09/(r**3)
w_tile = np.tile(w, (len_t, 1)).T
t_tile = np.tile(np.abs(t), (len_r, 1))
a = 3 * w_tile * t_tile
q = np.sqrt(2*a/np.pi)
ss, cs = fresnel(q)
cs = cs/q
ss = ss/q
ineq0 = q == 0.0
cs[ineq0] = 0
ss[ineq0] = 0
basis = cs * np.cos(w_tile*t_tile) + ss * np.sin(w_tile*t_tile)
basis[ineq0] = 1
basis = basis.T

# create arrays for saving the old simulated dist and deer for overlays
dist_old = np.zeros(len_r)
deer_old = np.zeros(len_t)
# create simulation parameters will be considered global variables
amp1 = 1
cen1 = 25.0
wid1 = 2.0
amp2 = 1
cen2 = 35.0
wid2 = 3.0


def load_data(idx=1):
    """Read in B1 data from the directory Data\B.

    Keyword argument:
    idx -- the number of the training data set to read in
    
    Returns:
    The distance and deer data used in training and
    the four simulations termed 1A, 1B, 2A, and 2B.    
    """    
    deer = np.genfromtxt(r'.\Data\B\B1_deer_{:05.0f}.dat'.format(idx),delimiter=',')
    dist = np.genfromtxt(r'.\Data\B\B1_dist_{:05.0f}.dat'.format(idx),delimiter=',')
    sim1A = np.genfromtxt(r'.\Data\B\B1_sim1A_{:05.0f}.dat'.format(idx),delimiter=',')
    sim1B = np.genfromtxt(r'.\Data\B\B1_sim1B_{:05.0f}.dat'.format(idx),delimiter=',')
    sim2A = np.genfromtxt(r'.\Data\B\B1_sim2A_{:05.0f}.dat'.format(idx),delimiter=',')
    sim2B = np.genfromtxt(r'.\Data\B\B1_sim2B_{:05.0f}.dat'.format(idx),delimiter=',')
    
    return deer,dist,sim1A,sim1B,sim2A,sim2B


def create_sim_graphs(r,dist,t,deer):
    """Will create a 1x2 plot with the simulated distance distance distribtuion
    on the left and the resultant deer decay on the right. It will ovelay the
    previous calculation.

    Input argument:
        r -- the distance distribution range
        dist -- the simulated distance distribution
        t -- the time scale for the deer decay
        deer -- the simulated deer decay
    
    Returns:
        graphs -- a single Plotly graph figure
    """
    global dist_old, deer_old
    
    trace_dist_old = Scatter(x=r,y=dist_old,showlegend=True, name='previous',
                     line = dict(color= 'red', width = 2) )
    trace_dist = Scatter(x=r,y=dist,showlegend=True, name='current',
                     line = dict(color= 'blue', width = 2) )
    trace_deer_old = Scatter(x=t,y=deer_old,showlegend=False, name='previous',
                     line = dict(color= 'red', width = 2) )
    trace_deer = Scatter(x=t,y=deer,showlegend=False, name='current',
                     line = dict(color= 'blue', width = 2) )
    fig = tools.make_subplots(rows=1, cols=2,
                subplot_titles=('Simulated Distribution', 'Simulated Deer Decay')) 
    fig.append_trace(trace_dist_old, 1, 1)
    fig.append_trace(trace_dist, 1, 1)
    fig.append_trace(trace_deer_old, 1, 2)
    fig.append_trace(trace_deer, 1, 2)
    fig['layout']['xaxis1'].update(title='r(Å)')
    fig['layout']['yaxis1'].update(title='P(r)',showticklabels=False)
    fig['layout']['xaxis2'].update(title='Time(µs)')
    fig['layout']['yaxis2'].update(title='Echo')
    
    graphs = [ {
                'data': fig.data,
                'layout': fig.layout
               }]
 
    return graphs
    

def create_train_exp_graphs(deer,dist,sim1A,sim1B,sim2A,sim2B):
    """Will create a 2x3 plot with the deer data on the left and the distance
    distribution and the four simulations in a grid.
    This is created as a one figure subplot that will then be rendered full width
    in th browser.

    Keyword arguments:
        deer -- x and y data for the training deer data (input)
        dist -- x and y data for the training distance data (target)
        simXX -- the simulations from the four models
    
    Returns:
        graphs -- a single Plotly graph figure
    """     
    traced = Scatter(x=deer[0],y=deer[1],showlegend=False,
                     line = dict(color= 'blue', width = 2) )
    trace0 = Scatter(x=dist[0],y=dist[1],showlegend=False,
                     line = dict(color= 'blue', width = 2) )
    trace1A = Scatter(x=dist[0],y=sim1A,showlegend=False,
                      line = dict(color= 'orange', width = 2) )
    trace1B = Scatter(x=dist[0],y=sim1B,showlegend=False,
                      line = dict(color= 'purple', width = 2) )
    trace2A = Scatter(x=dist[0],y=sim2A,showlegend=False,
                      line = dict(color= 'green', width = 2) )
    trace2B = Scatter(x=dist[0],y=sim2B,showlegend=False,
                      line = dict(color= 'red', width = 2) )
    
    fig = tools.make_subplots(rows=2, cols=3, 
                subplot_titles=('Input Deer Decay', 'Model 1A', 'Model 1B', '', 'Model 2A','Model 2B'))
    fig.append_trace(traced, 1, 1)
    fig.append_trace(trace0, 1, 2)
    fig.append_trace(trace1A, 1, 2)
    fig.append_trace(trace0, 1, 3)
    fig.append_trace(trace1B, 1, 3)
    fig.append_trace(trace0, 2, 2)
    fig.append_trace(trace2A, 2, 2)
    fig.append_trace(trace0, 2, 3)
    fig.append_trace(trace2B, 2, 3)
    fig['layout']['yaxis2'].update(title='P(r)',showticklabels=False)
    fig['layout']['yaxis3'].update(title='P(r)',showticklabels=False)
    fig['layout']['yaxis5'].update(title='P(r)',showticklabels=False)
    fig['layout']['yaxis6'].update(title='P(r)',showticklabels=False)
    fig['layout']['xaxis5'].update(title='r(Å)')
    fig['layout']['xaxis6'].update(title='r(Å)')
    
    graphs = [ {
                'data': fig.data,
                'layout': fig.layout
               }]
 
    return graphs


def plots(arg_1,*args):
    """This will take in the data to be plotted and create Javascript

    Keyword arguments:
        ... -- the data read in by load_data
    
    Returns:
        ids -- the ids of the javascript convertly Plotly figure
        graphJSON -- the javascript for Plotly figure
    """    
    # for training plots args are: deer,dist,sim1A,sim1B,sim2A,sim2B
    # for simulation plots args are: r,g_dist,t,deer_sim; old are global
    
    # create visuals
    if arg_1 == 'sim':
        graphs = create_sim_graphs(*args)
    elif arg_1 == 'train_exp':
        graphs = create_train_exp_graphs(*args)
    else:
        return
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return ids, graphJSON


# index is initial webpage with starting visuals
@app.route('/')
@app.route('/index')
def index():
    return render_template('home.html')


# runs the initial page for the training examples page
@app.route('/train_setup')
def train_setup():
    # load data
    deer,dist,sim1A,sim1B,sim2A,sim2B = load_data(24956)
    ids, graphJSON = plots('train_exp',deer,dist,sim1A,sim1B,sim2A,sim2B)

    return render_template('train_exp.html', ids=ids, graphJSON=graphJSON)


# runs the training examples page
@app.route('/train')
def train():
    # load data
    # save user input in query
    query = request.args.get('query', '')
    try:
        int(query)
    except:
        query = 24956
    query = int(query)
    if abs(query) > 30000 or query == 0:
        query = 24956
    deer,dist,sim1A,sim1B,sim2A,sim2B = load_data(query)
    ids, graphJSON = plots('train_exp',deer,dist,sim1A,sim1B,sim2A,sim2B)

    return render_template('train_exp.html', ids=ids, graphJSON=graphJSON)


# runs the initial page for the simulations page
@app.route('/sim_setup')
def sim_setup():
    return render_template('simulate.html', 
                           amp1=amp1, cen1=cen1, wid1=wid1,
                           amp2=amp2, cen2=cen2, wid2=wid2)


# takes input and updates the webpage
@app.route('/sim_run')
def sim_run():
    # need to make this global so the value can be continually updated
    global dist_old, deer_old, amp1, cen1, wid1, amp2, cen2, wid2

    amp1 = np.float(request.args.get('amp1'))
    cen1 = np.float(request.args.get('cen1'))
    wid1 = np.float(request.args.get('wid1'))
    amp2 = np.float(request.args.get('amp2'))
    cen2 = np.float(request.args.get('cen2'))
    wid2 = np.float(request.args.get('wid2'))
    
    g_dist = amp1 * norm.pdf(r, cen1, wid1) + amp2 * norm.pdf(r, cen2, wid2)
    g_dist = g_dist / np.sum(g_dist)
  
    deer_sim = np.matmul(basis,g_dist)
    
    ids, graphJSON = plots('sim',r,g_dist,t*1e6,deer_sim)
    
    deer_old = deer_sim
    dist_old = g_dist
    
    return render_template('simulate.html', ids=ids, graphJSON=graphJSON,
                           amp1=amp1, cen1=cen1, wid1=wid1,
                           amp2=amp2, cen2=cen2, wid2=wid2)
   

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()