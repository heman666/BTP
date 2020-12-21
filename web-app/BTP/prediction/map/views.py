from django.shortcuts import render
import requests,json
import folium
import geopandas,os
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests,json
import numpy as np
import time,io,urllib,base64
import pandas as pd
import matplotlib.pyplot as plt
import random
import calendar
import datetime as DT
import random

import networkx as nx
import datetime as dt
import pickle
import os

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter("ignore")


gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')



class GCNLSTM(nn.Module):
    def __init__(self, n_feats, seq_len):
        super(GCNLSTM, self).__init__()
        self.n_feats = n_feats
        self.seq_len = seq_len
        self.n_hidden = 6 # number of hidden states for LSTM cell
        self.n_layers = 5 # number of stacked LSTM layers, original 3, new 5

        self.lstm = nn.LSTM(input_size=n_feats,
                            hidden_size=self.n_hidden,
                            num_layers=self.n_layers,
                            batch_first=True,
                            # bidirectional=True,
                    dropout=0.3)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.lstm(h)[0]

class GCNLinear(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLinear, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.layer0 = GCNConv(124,6,1,20)
        self.layer1 = GCNLSTM(1, 6)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = GCNLinear(36, 150)
        # self.dropout2 = nn.Dropout(0.2)
        self.layer3 = GCNLinear(150, 50)
        self.layer4 = GCNLinear(50, 1)

    def forward(self, g, features):
        batch_size, seq_len, n_feats = features.size()
        # x = self.layer0(g,features)
        x = self.layer1(g, features)
        x = x.contiguous().view(batch_size, -1) # flatten
        x = F.relu(self.layer2(g, x))
        x = F.relu(self.layer3(g, x))
        x = F.relu(self.layer4(g, x))
        return x

def create_model():
    th.manual_seed(0)
    net = Net()
    net = net.train()
    print("########################################## NET #####################################")
    print(net)
    print("####################################################################################")
    return net


def get_data():
    latlong = pd.read_csv('./latlong.csv')
    latlong.sort_values(by='Station',inplace=True)
    latlong.reset_index(inplace=True)
    RAIN = []

    y = np.random.randn(124,1)
    for i in range(len(latlong)):
        name = latlong.iloc[i,:]['Station']
        latitude = latlong[latlong['Station'] == name ]['Latitude'][i]
        longitude = latlong[latlong['Station'] == name ]['Longitude'][i]
        # url = "http://api.weatherapi.com/v1/current.json?key=46762d8f7db04acfaa4132105200610&q={},{}".format(latitude,longitude)
        # r = requests.get(url)
        # data = r.json()
        # print(data)
        # rainfall = data["current"]["precip_mm"]
        RAIN.append(y[i][0])

    return RAIN

def home(request):
    net = create_model()
    chkpnt = th.load("./network_model_not_normalized_200epochs_4neighbors.pth",map_location=th.device('cpu'))
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    net.load_state_dict(chkpnt['model_state_dict'])
    optimizer.load_state_dict(chkpnt['optimizer_state_dict'])
    # epoch = chkpnt['epoch']
    with open("./direct_graph.pkl", 'rb') as f:
        g = pickle.load(f)

    g = dgl.from_networkx(g)
    print("This is my graph : {}".format(g))

    net.eval()

    rainfall_data = pd.read_csv("./Final-Data/rainfall.csv",index_col="Date")
    rain_data = rainfall_data.iloc[-6:,:].T.values
    rain_data = np.reshape(rain_data,(rain_data.shape[0],6,1))
    print("Shape : {}".format(rain_data.shape))
    pred = net(g,th.Tensor(rain_data))
    # print(pred)

    data = new_func(rain_data,pred)
    # print(data)

    today = DT.date.today()
    date = []
    week_name = []
    for i in range(7):
        tod = today.strftime('%d/%m/%Y')
        date.append(tod)
        week_name.append(calendar.day_name[today.weekday()])
        today = today + DT.timedelta(days=1)
    print(date)
    # print(rain_data)
    RAIN = rain_data[-6]
    RAIN = np.reshape(RAIN,(RAIN.shape[0],)).tolist()
    for i in range(len(RAIN)):
        RAIN[i] = round(RAIN[i],2)

    pred = pred.detach().numpy()
    RAIN.append(round(pred[-6][0],2))
    # print(RAIN)
    # print(date)
    # print(week_name)
    temp = []
    for x,y,z in zip(date,week_name,RAIN):
        temp1 = []
        temp1.append(x)
        temp1.append(y)
        temp1.append(z)
        temp.append(list(temp1))

    print(temp)
    context = {
        "data" : data,
        "temp" : temp,
        "date" : date,
        "week_name" : week_name,
        "rain" : RAIN
    }

    return render(request,'map/home_new.html',context)


def new_func(rain_data,pred):
    final_list = {}
    final_list["type"] = "FeatureCollection"
    final_list["features"] = []

    latlong = pd.read_csv('./latlong.csv')
    latlong.sort_values(by='Station',inplace=True)
    latlong.reset_index(inplace=True)
    print(pred.size())
    pred = pred.detach().numpy()
    # y = np.random.randn(123,1)
    for i in range(len(latlong)):
        features = {}
        features['type'] = "Feature"
        properties = {}
        properties['name'] = latlong.iloc[i,:]['Station']
        properties['latitude'] = latlong[latlong['Station'] == properties['name']]['Latitude'][i]
        properties['longitude'] = latlong[latlong['Station'] == properties['name']]['Longitude'][i]

        rainfall = rain_data[i]
        rainfall = np.reshape(rainfall,(rainfall.shape[0],))
        rainfall = rainfall.tolist()
        rainfall.append(round(pred[i][0],2))
        for i in range(len(rainfall)):
            rainfall[i] = round(rainfall[i],2)
        # print(rainfall)
        properties['rainfall'] = list(rainfall)

        # print(random.uniform(0,1))
        properties['flood_probability'] = round(random.uniform(0,0.05),2)
        properties['drought_probability'] = round(random.uniform(0,0.05),2)
        properties['risk_estimation'] = round(random.uniform(0,0.05),2)


        features['properties'] = properties
        geometry = {}
        geometry['type'] = "Point"
        geometry["coordinates"] = [properties['longitude'],properties['latitude']]
        features['geometry'] = geometry
        final_list['features'].append(features)
    # print(final_list)
    return final_list


def folium_map(request):
    return render(request,'map/folium_map.html')

def mapex(request):
    return render(request,'map/maps_ex.html')

def index_map(request):
    return render(request , 'map/index.html')

def temp(request):
    return render(request , 'map/temp.html')

# def save_model():
#     filename = 'model.sav'
#     joblib.dump(model,filename)
#
# def load_model():
#     cls = joblib.load(filename)
