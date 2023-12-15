# -*- coding: utf-8 -*-
import numpy as np
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pymc.pytensorf import collect_default_updates
import pytensor
import geopandas as gpd
from scipy.spatial import distance_matrix

np.random.seed(27)

#####plotting parameters
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.titlesize': 12})
plt.rcParams['font.family'] = "DeJavu Serif"
plt.rcParams['font.serif'] = "Cambria Math"

###############################################################################

cpop = pd.read_csv("./data/countries_population.csv")

gdf = gpd.read_file("./data/shape_files/ne_110m_admin_0_countries.shp")
#gdf["SOVEREIGNT"] = gdf["SOVEREIGNT"].str.replace("United States of America", "United States")

gdf = gdf[gdf.BRK_NAME.isin(cpop.country)]

cpop = cpop[cpop.country.isin(gdf.BRK_NAME)]
cpop.reset_index(inplace=True, drop=True)

gdf = gdf[gdf.BRK_NAME.isin(cpop.country)]
gdf = gdf.sort_values("BRK_NAME")

coords = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])

xy = np.array([c[0] for c in coords])
D = distance_matrix(xy, xy)#*100 + 0.000000001 #distance matrix in meters

gdf['x'] = xy[:,0]
gdf['y'] = xy[:,1]

df1 = pd.read_csv("./data/country_flow_202001.csv", index_col=0)
df1 = df1[df1.index.isin(cpop.country)]
df1 = df1[df1.columns.intersection(df1.index)]
df2 = pd.read_csv("./data/country_flow_202002.csv", index_col=0)
df2 = df2[df2.index.isin(cpop.country)]
df2 = df2[df2.columns.intersection(df2.index)]
df3 = pd.read_csv("./data/country_flow_202003.csv", index_col=0)
df3 = df3[df3.index.isin(cpop.country)]
df3 = df3[df3.columns.intersection(df3.index)]

df = df1 + df2 + df3 #air travel mobility matrix

df = df.sort_index()
df = df.sort_index(axis=1)

M = df.values/90

#years = ["2018","2019","2020","2021","2022"]
years = ["2021","2022"]

df = pd.read_csv("./data/dengue_cases.csv", encoding = "ISO-8859-1")
df['year'] = [str(d[:4]) for d in df.calendar_start_date]

df = df[df.year.isin(years)]
bra = df[df["adm_0_name"]=="Brazil"]
bra = bra.sort_values("calendar_start_date")
bra.reset_index(inplace=True, drop=True)

df = df[df.year=="2020"]
df = df.groupby(["adm_0_name","year"], as_index=False).sum()

cpop = pd.read_csv("./data/countries_population.csv")
N = cpop[cpop['country']=="Brazil"].population.values[0]
initial_infected = bra.dengue_total.values[0]

weeks = len(bra)

infected_obs = bra.dengue_total.values

obs = (infected_obs-infected_obs.mean())/infected_obs.std() #standarised obs

d = D / (np.max(D, axis=1) ) + 0.00001

coords = {'origin':np.arange(M.shape[0]), 'destination':np.arange(M.shape[0])}

with pm.Model(coords=coords) as mod:
    tau = pm.HalfNormal("tau", 1)
    omega = pm.HalfNormal("omega", 1) #travel decay prior
    eta = pm.HalfNormal("eta", 10) #distance decay prior
    theta = pm.HalfNormal("theta", 10, shape=M.shape) #dims=('origin', 'destination'))
    mu = pm.Deterministic("mu", ((theta - pt.exp(-eta*D))/omega )**tau )
    y = pm.Poisson("y", mu=mu, observed=M)

with mod:
    ppc = pm.sample_prior_predictive(var_names=['y'])


with mod:
    idata = pm.sample()

# #connectivity model (sample from prior)
# with pm.Model() as mod:
#     #mu = pm.HalfNormal("mu", 1, shape=M.shape)
#     omega = pm.HalfNormal("omega", 1) #travel decay prior
#     tau = pm.HalfNormal("tau", 1) #travel decay prior
#     eta = pm.HalfNormal("eta", 1) #distance decay prior
#     psi = pm.Deterministic("psi", pt.exp(-eta*D) / pt.max(pt.exp(-eta*D) ) ) #distance decay prob
#     p = pm.Deterministic("p", (M**tau / pt.sum(M**tau, axis=1)) ) #travel probability
#     theta = pm.Deterministic("theta", psi + omega*p) #connectivity score


# # Connectivity model (works okay)
# with pm.Model() as mod:
#     delta = pm.Normal("delta", 0, 1, shape=M.shape) # mobility prior
#     tau = pm.HalfNormal("tau", 1) #non-linear scaling
#     omega = pm.HalfNormal("omega", 1)#topological term contribution
#     mu = pm.Deterministic("mu", pt.exp(delta)) #mobility estimate
#     eta = pm.HalfNormal("eta", 1) #distance decay prior
#     psi = pm.Deterministic("psi", pt.exp(-eta*D) / pt.max(pt.exp(-eta*D) ) ) #distance decay prob
#     p = pm.Deterministic("p", (mu**tau / pt.max(mu**tau)) ) #travel probability
#     theta = pm.Deterministic("theta", psi + omega*p) #connectivity score
#     y = pm.Poisson("y", mu=mu, observed=M)
    
# with mod:
#     ppc = pm.sample_prior_predictive(100, var_names=["y"])    
    
# az.plot_ppc(ppc, group="prior", kind="cumulative")

with mod:
    idata = pm.sample()

theta_pos = az.extract(idata)['theta'].values
theta_m = theta_pos.mean(axis=2)

mu_pos = az.extract(idata)['mu'].values
mu_m = mu_pos.mean(axis=2)

psi_pos = az.extract(idata)['psi'].values
psi_m = psi_pos.mean(axis=2)

p_pos = az.extract(idata)['p'].values
p_m = p_pos.mean(axis=2)

# with mod:
#     pred = pm.sample_posterior_predictive(idata)

# ppred = az.extract(pred, group="posterior_predictive")['y'].values

# pmean = ppred.mean(axis=2)

# samps = np.random.randint(0,4000, 100)
# for s in samps:
#     plt.plot(ppred.sum(axis=1)[:,s], color="g", alpha=0.2)
# plt.plot(pmean.sum(axis=1))
# plt.plot(M.sum(axis=1))




