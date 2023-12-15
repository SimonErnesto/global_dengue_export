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

#Mi = np.array([M for i in range(months)])

# eta.append(pm.HalfNormal("eta"+str(i), 1))
# rho.append(pm.HalfNormal("rho"+str(i), 1)) #space scale
# K = eta[i]**2 * pm.gp.cov.ExpQuad(1, rho[i])
# gp = pm.gp.Latent(cov_func=K)

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
    
    
    
with mod:
    ppc = pm.sample_prior_predictive(1000, var_names=["y"])    

az.plot_ppc(ppc, group="prior", kind="cumulative", var_names='y')

ppc_y = az.extract(ppc, group="prior_predictive")['y'].values

df['y_m'] = ppc_y.mean(axis=1).round(0)
df['y_s'] = ppc_y.std(axis=1)
df['y_u'] = ppc_y.mean(axis=1) + ppc_y.std(axis=1)
df['y_l'] = ppc_y.mean(axis=1) - ppc_y.std(axis=1)
df['y_l'][df['y_l'] < 0] = 0 

df_year = df[['month','infected','y_m','y_s','y_u','y_l']].groupby(["month"], as_index=False).sum()
df_country = df.groupby(["country","abbrev"], as_index=False).sum()

fig, ax = plt.subplots(2,1, figsize=(20,6))
ax[0].plot(np.arange(len(df_year)), df_year.infected, color='k', label="Observed")
ax[0].plot(np.arange(len(df_year)), df_year.y_m, color='crimson', linestyle="--", label="Prior predictive mean")
ax[0].fill_between(np.arange(len(df_year)), df_year.y_l, df_year.y_u, color='crimson', alpha=0.2, label="Prior predictive SD")
ax[0].legend()
ax[0].set_ylabel("Count", size=16)
ax[0].set_xticks(np.arange(len(df_year)), month0, rotation=90)
ax[0].set_xlabel("Month")
ax[0].set_title("Prior Predictions Time (2020)")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].grid(alpha=0.2)
ax[1].plot(np.arange(len(df_country)), df_country.infected, color='k', label="Observed")
ax[1].plot(np.arange(len(df_country)), df_country.y_m, color='crimson', linestyle="--", label="Prior predictive mean")
ax[1].fill_between(np.arange(len(df_country)), df_country.y_l, df_country.y_u, color='crimson', alpha=0.2, label="Prior predictive SD")
ax[1].legend()
ax[1].set_xticks(np.arange(len(df_country)), df_country.abbrev, rotation=90)
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Count")
ax[1].set_title("Prior Predictions Space (2020)")
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig("prior_predictives.png", dpi=300)
plt.show()
plt.close()


with mod:
    idata = pm.sample(2000, tune=2000, chains=4, target_accept=0.95, nuts_sampler="numpyro")

az.to_netcdf(idata, "dengue_import_model_idata.nc")

with mod:
    pred = pm.sample_posterior_predictive(idata)

pred_y = az.extract(pred, group="posterior_predictive")['y'].values

df['y_m'] = pred_y.mean(axis=1).round(0)
df['y_s'] = pred_y.std(axis=1)
df['y_u'] = pred_y.mean(axis=1) + pred_y.std(axis=1)
df['y_l'] = pred_y.mean(axis=1) - pred_y.std(axis=1)
df['y_l'][df['y_l'] < 0] = 0 

df_year = df[['month','infected','y_m','y_s','y_u','y_l']].groupby(["month"], as_index=False).sum()
df_country = df.groupby(["country","abbrev"], as_index=False).sum()

fig, ax = plt.subplots(2,1, figsize=(20,6))
ax[0].plot(np.arange(len(df_year)), df_year.infected, color='k', label="Observed")
ax[0].plot(np.arange(len(df_year)), df_year.y_m, color='purple', linestyle="--", label="Posterior predictive mean")
ax[0].fill_between(np.arange(len(df_year)), df_year.y_l, df_year.y_u, color='purple', alpha=0.2, label="Posterior predictive SD")
ax[0].legend()
ax[0].set_ylabel("Count", size=16)
ax[0].set_xticks(np.arange(len(df_year)), month0, rotation=90)
ax[0].set_xlabel("Month")
ax[0].set_title("Posterior Predictions Time (2020)")
ax[0].spines[['right', 'top']].set_visible(False)
ax[0].grid(alpha=0.2)
ax[1].plot(np.arange(len(df_country)), df_country.infected, color='k', label="Observed")
ax[1].plot(np.arange(len(df_country)), df_country.y_m, color='purple', linestyle="--", label="Posterior predictive mean")
ax[1].fill_between(np.arange(len(df_country)), df_country.y_l, df_country.y_u, color='purple', alpha=0.2, label="Posterior predictive SD")
ax[1].legend()
ax[1].set_xticks(np.arange(len(df_country)), df_country.abbrev, rotation=90)
ax[1].set_xlabel("Country")
ax[1].set_ylabel("Count")
ax[1].set_title("Posterior Predictions Space (2020)")
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig("posterior_predictives.png", dpi=300)
plt.show()
plt.close()


az.plot_trace(idata, var_names=["eta","tau","theta","rho","phi"], kind="rank_vlines")
plt.tight_layout()
plt.savefig("trace_plot")

  
p = az.extract(idata)['psi'].values
p_mean = p.mean(axis=3)
p_df0 = pd.DataFrame(p_mean[0], columns=df.country.unique(), index=df.country.unique())

summ = az.summary(idata, hdi_prob=0.9)
summ.to_csv("idata_summary.csv")

summ_small = az.summary(idata, hdi_prob=0.9, var_names=["eta","tau","theta","rho","phi"])
summ_small.to_csv("small_summary.csv")

# lam_pos = az.extract(idata)['lam'].values
# lam = lam_pos.mean(axis=1)
# P = lam/Ni
# P = P.reshape([months,countries])

# psi = np.array([(P[i] * M) / (P[i] * M).sum(axis=0)  for i in range(months)])
