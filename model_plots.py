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
import xarray as xr
from tqdm import tqdm
import calendar

np.random.seed(27)
rng = np.random.default_rng(27)

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
gdf = gdf[gdf.NAME_EN != "Antarctica"]

cpop = cpop[cpop.country.isin(gdf.BRK_NAME)]
cpop.reset_index(inplace=True, drop=True)

gdf = gdf[gdf.BRK_NAME.isin(cpop.country)]
gdf = gdf.sort_values("BRK_NAME")

coords = gdf['geometry'].apply(lambda x: x.representative_point().coords[:])

xy = np.array([c[0] for c in coords])
D = distance_matrix(xy, xy)*100 + 0.000000001 #distance matrix in meters

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

Mdf = df1 + df2 + df3 #air travel mobility matrix

Mdf = Mdf.sort_index()
Mdf = Mdf.sort_index(axis=1)

M = Mdf.values/3

df = pd.read_csv("./data/dengue_cases.csv", encoding = "ISO-8859-1")
df = df[["adm_0_name","calendar_start_date","dengue_total"]]
df = df.rename(columns={"adm_0_name":"country","dengue_total":"infected"})
df['year'] = [str(d[:4]) for d in df.calendar_start_date]
labels = {"Lao People's Democratic Republic":"Laos", "French Guiana":"Guyana", "Hong Kong":"China"}
df = df.replace(labels)
df = df[df.country.isin(cpop.country.unique())]
df = df[df.year.isin(["2020"])]
df['month'] = [str(d[5:7]) for d in df.calendar_start_date]
df = df.groupby(["country","year","month"], as_index=False).sum()
keeps = [c for c in df.country.unique() if len(df[df.country==c]) > df.month.unique().shape[0]-1]
df = df[df.country.isin(keeps)]
df = df.drop(["calendar_start_date"], axis=1)
df.reset_index(inplace=True, drop=True)

dengue_countries = df.country.unique()

months = len(df.month.unique())

dfc = pd.concat([cpop for m in range(months)])
dfc = dfc.drop(['c_number','population'],axis=1)
dfc['month'] = np.repeat(df.month.unique(), len(cpop))
dfc = dfc.sort_values(["country", "month"])

dfc = dfc[~dfc.country.isin(df.country.unique())]

df = pd.concat([dfc,df])

df = df.sort_values(['country', 'month'])
df.reset_index(inplace=True, drop=True)

df = df.fillna(0)

countries = len(df.country.unique())

abbrev = np.concatenate([np.repeat(gdf.ABBREV.values[c], months) for c in range(countries) ])

month0 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
month = np.concatenate([month0 for c in range(countries)])

df['month'] = month
df['abbrev'] = abbrev

df['lon'] = np.concatenate([xy[:,0] for m in range(months)])
df['lat'] = np.concatenate([xy[:,1] for m in range(months)])


df_obs = df[df.country.isin(dengue_countries)]
df_uns = df[~df.country.isin(dengue_countries)]

dfc = df_obs.drop_duplicates("country")
xy_obs = np.array([dfc.lon.values, dfc.lat.values]).T
D_obs = distance_matrix(xy_obs, xy_obs)*100 + 0.000000001

T = np.arange(months)
# T = np.arange(months)
# T = np.array([T,T])
# T = distance_matrix(T.T, T.T)

coords = {'country':df.abbrev.unique(), 'month':df.month.unique(),"feature":["longitude", "latitude"]}

c_idx = pd.factorize(df_obs.country)[0]
t_idx = pd.factorize(df_obs.month)[0]

dengue_idx = cpop[cpop.country.isin(dengue_countries)].index.values

obs = df_obs.infected.values

infected  = df.infected.values.astype("int")
t_idx = pd.factorize(df.month)[0]
c_idx = pd.factorize(df.country)[0]

df = df.merge(cpop, on="country")

Ni = Nj = df.population.values

N = df.population.unique()

Mi = np.repeat(M.sum(axis=1), months)
Mj = np.repeat(M.sum(axis=0), months)


#### Gaussian process model
with pm.Model(coords=coords) as mod:
    
    eta = pm.HalfNormal("eta", 1)
    tau = pm.HalfNormal("tau", 1) #time scale
    K = eta**2 * pm.gp.cov.ExpQuad(1, tau**2)
    gpt = pm.gp.Latent(cov_func=K) #gp time
    a = gpt.prior("a", X=T[:,None], dims="month")
    
    theta = pm.HalfNormal("theta", 1)
    rho = pm.HalfNormal("rho", 1) #space scale
    C = theta**2 * pm.gp.cov.ExpQuad(2, rho**2)
    gps = pm.gp.Latent(cov_func=C) #gp space
    b = gps.prior("b", X=xy)
    
    lam = pm.Deterministic("lam", pt.exp(a[t_idx] + b[c_idx])) 

    P = lam / Ni
    P = P.reshape([months,countries])

    psi = pt.as_tensor_variable([(P[i] * M).T / (P[i] * M).sum(axis=1)  for i in range(months)])
    
    psi = pm.Deterministic("psi", psi)
    
    phi = pm.HalfNormal("phi", 1)
    
    y = pm.NegativeBinomial("y", mu=lam, alpha=phi, observed=infected)
    
    
idata = az.from_netcdf("dengue_import_model_idata.nc")


psi = az.extract(idata)['psi'].values
p_mean = psi.mean(axis=3)
p_sd = psi.std(axis=3)
    


fig, ax = plt.subplots(3, 4, figsize=(12,10))
for m in tqdm(range(months)):
    if m < 9:
        num = "0"+str(m+1)
    else:
        num = str(m+1)
    mapcolor = "tan"
    color = "steelblue"
    if m < 4:
        l = m
        u = 0
    if m in [4,5,6,7]:
        l = m-4
        u = 1
    if m > 7:
        l = m-8
        u=2
    im = ax[u,l].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='Blues', origin="lower", alpha=0.6, vmin=0,vmax=1)
    im.set_visible(False)
    gdf.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax[u,l])
    for i in range(countries):
        for j in range(countries):
            if j != i:
                p1 = [gdf['x'].values[i], gdf['x'].values[j]]
                p2 = [gdf['y'].values[i], gdf['y'].values[j]]
                #x,y = draw_curve(p1,p2)
                w = p_mean[m,i,j] 
                ax[u,l].plot(p1,p2, color=color, alpha=0.6*w, linewidth=0.5)
    cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label(label="Probability", size=8)
    ax[u,l].axis("off")
    ax[u,l].set_title(month[m], size=12)
fig.subplots_adjust(top=0.2)
plt.suptitle("2020 Infected Passengers Flow (posterior standard deviation)", size=16)
plt.subplots_adjust(wspace=2)
plt.tight_layout()
plt.savefig("./world_plots/plots_stds.png", dpi=300)
plt.close()


fig, ax = plt.subplots(3, 4, figsize=(12,10))
for m in tqdm(range(months)):
    if m < 9:
        num = "0"+str(m+1)
    else:
        num = str(m+1)
    mapcolor = "tan"
    color = "forestgreen"
    if m < 4:
        l = m
        u = 0
    if m in [4,5,6,7]:
        l = m-4
        u = 1
    if m > 7:
        l = m-8
        u=2
    im = ax[u,l].imshow(np.arange(4, 0, -1).reshape(2, 2)/4, cmap='Greens', origin="lower", alpha=0.6, vmin=0,vmax=1)
    im.set_visible(False)
    gdf.plot(color=mapcolor, edgecolor="k", linewidth=0.1, ax=ax[u,l])
    for i in range(countries):
        for j in range(countries):
            if j != i:
                p1 = [gdf['x'].values[i], gdf['x'].values[j]]
                p2 = [gdf['y'].values[i], gdf['y'].values[j]]
                #x,y = draw_curve(p1,p2)
                w = p_sd[m,i,j] 
                ax[u,l].plot(p1,p2, color=color, alpha=0.6*w, linewidth=0.5)
    cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, pad=0.01)
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label(label="Probability", size=8)
    ax[u,l].axis("off")
    ax[u,l].set_title(month[m], size=12)
fig.subplots_adjust(top=0.2)
plt.suptitle("2020 Infected Passengers Flow (posterior standard deviation)", size=16)
plt.subplots_adjust(wspace=2)
plt.tight_layout()
plt.savefig("./world_plots/plots_stds.png", dpi=300)
plt.close()


year_prob_mean = pd.DataFrame(p_mean.mean(axis=0), columns=df.country.unique(), index=df.country.unique())
year_prob_mean.to_csv("year_prob_mean.csv")

year_prob_sd = pd.DataFrame(p_sd.mean(axis=0), columns=df.country.unique(), index=df.country.unique())
year_prob_sd.to_csv("year_prob_sd.csv")


