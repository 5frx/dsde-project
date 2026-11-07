"""Reproducible Component 2 audit and PM2.5 analysis.
Usage: python analyze_pm25.py --csv data.csv --out output
Reads the supplied prepared CSV; performs statistical analysis only.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VEHICLES=['car_count','motorcycle_count','bus_count','truck_count']
WEATHER=['temperature_2m','precipitation','wind_speed_10m','wind_direction_10m','relative_humidity_2m','cloud_cover','surface_pressure']

def run(src,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 df=pd.read_csv(src)
 required=['camera_id','hourly_timestamp','pm25','sensor_id','station_distance_m','is_fallback',*VEHICLES]
 missing=[c for c in required if c not in df];assert not missing, f'Missing columns: {missing}'
 df['local_time']=pd.to_datetime(df['hourly_timestamp'],errors='coerce')
 df['total_vehicles']=df[VEHICLES].sum(axis=1)
 grp=['camera_id','hourly_timestamp']
 consistency=df.groupby(grp).agg(pm25_values=('pm25','nunique'),source_values=('sensor_id','nunique')).reset_index()
 ch=df.groupby(grp,as_index=False).agg({**{v:'mean' for v in VEHICLES+['total_vehicles']},**{v:'first' for v in ['pm25','sensor_id','location_id','station_distance_m','is_fallback',*WEATHER] if v in df}})
 ch['local_time']=pd.to_datetime(ch['hourly_timestamp']);ch['date']=ch.local_time.dt.strftime('%Y-%m-%d');ch['hour_of_day']=ch.local_time.dt.hour
 ch['total_vehicles']=ch[VEHICLES].sum(axis=1)
 audit={'image_rows':len(df),'camera_hours':len(ch),'cameras':int(df.camera_id.nunique()),'stations':int(df.sensor_id.nunique()),'pm25_valid_images':int(df.pm25.notna().sum()),'pm25_valid_camera_hours':int(ch.pm25.notna().sum()),'pm25_missing_images':int(df.pm25.isna().sum()),'zero_pm25_camera_hours':int((ch.pm25==0).sum()),'fallback_share_pm25_camera_hours':float(ch.loc[ch.pm25.notna(),'is_fallback'].mean()),'max_station_distance_m':float(ch.station_distance_m.max()),'pm25_conflicts_within_camera_hour':int((consistency.pm25_values>1).sum()),'sensor_conflicts_within_camera_hour':int((consistency.source_values>1).sum()),'start':str(df.local_time.min()),'end':str(df.local_time.max()),'unique_station_utc_hours':int(df[['sensor_id','hour']].drop_duplicates().shape[0]) if 'hour' in df else None}
 (out/'audit.json').write_text(json.dumps(audit,indent=2))
 ch.to_csv(out/'camera_hours.csv',index=False)
 # Dashboard-ready datasets with one observation per camera and hour.
 dash=out/'dashboard';dash.mkdir(parents=True,exist_ok=True)
 coords=df.groupby('camera_id',as_index=False).agg({col:'first' for col in ['latitude','longitude'] if col in df.columns})
 cameras=ch.groupby('camera_id',as_index=False).agg(
  image_hours=('hourly_timestamp','size'),
  avg_vehicles_per_frame=('total_vehicles','mean'),
  avg_car_count=('car_count','mean'),
  avg_motorcycle_count=('motorcycle_count','mean'),
  avg_bus_count=('bus_count','mean'),
  avg_truck_count=('truck_count','mean'),
  avg_pm25=('pm25','mean'),
  pm25_hours=('pm25','count'),
  avg_station_distance_m=('station_distance_m','mean'),
  fallback_share=('is_fallback','mean'))
 if len(coords.columns)>1: cameras=cameras.merge(coords,on='camera_id',how='left')
 cameras.to_csv(dash/'cameras.csv',index=False)
 dashboard_hours=ch.copy()
 dashboard_hours['pm25_available']=dashboard_hours.pm25.notna()
 dashboard_hours.to_csv(dash/'camera_hours.csv',index=False)
 hour_grid=ch.groupby(['camera_id','hour_of_day'],as_index=False).agg(
  avg_vehicles_per_frame=('total_vehicles','mean'),avg_pm25=('pm25','mean'),
  camera_hours=('hourly_timestamp','size'))
 hour_grid.to_csv(dash/'camera_hourly_profiles.csv',index=False)
 valid=ch.dropna(subset=['pm25']).copy()
 valid['log_total']=np.log1p(valid['total_vehicles'])
 candidates=['total_vehicles',*VEHICLES,*[c for c in WEATHER if c in valid]]
 correlations=[]
 for col in candidates:
  pair=valid[['pm25',col]].dropna()
  if len(pair)<30 or pair[col].nunique()<2:continue
  rho,p=spearmanr(pair[col],pair.pm25)
  correlations.append({'variable':col,'n_camera_hours':len(pair),'spearman_rho':float(rho),'p_uncorrected':float(p)})
 c=pd.DataFrame(correlations)
 if len(c):c['q_bh']=multipletests(c.p_uncorrected,method='fdr_bh')[1];c=c.sort_values('spearman_rho',key=abs,ascending=False)
 c.to_csv(out/'camera_hour_spearman.csv',index=False)
 c.to_csv(dash/'correlations.csv',index=False)
 # Aggregating by station and UTC hour avoids treating identical station readings as independent PM2.5 samples.
 station=valid.groupby(['sensor_id','hour'] if 'hour' in valid.columns else ['sensor_id','hourly_timestamp'],as_index=False).agg(pm25=('pm25','first'),total_vehicles=('total_vehicles','mean'),cameras=('camera_id','nunique'))
 station.to_csv(out/'station_hour_summary.csv',index=False)
 station.to_csv(dash/'station_hours.csv',index=False)
 # Two designs: a transparent unadjusted association and a conditional model.
 # Cluster standard errors by *PM station*, since outcome readings are shared across camera observations.
 formula='pm25 ~ log_total + C(camera_id) + C(hour_of_day) + C(date)'
 for v in ['temperature_2m','precipitation','wind_speed_10m','relative_humidity_2m','cloud_cover','surface_pressure']:
  if v in valid:formula+=' + '+v
 model_vars=['pm25','log_total','camera_id','hour_of_day','date','sensor_id']+[v for v in WEATHER if v in formula]
 mdf=valid.dropna(subset=model_vars).copy();mdf['camera_id']=mdf.camera_id.astype(str);mdf['sensor_id']=mdf.sensor_id.astype(str)
 model=smf.ols(formula,mdf).fit(cov_type='cluster',cov_kwds={'groups':mdf.sensor_id})
 keys=['log_total']+[v for v in ['temperature_2m','precipitation','wind_speed_10m','relative_humidity_2m','cloud_cover','surface_pressure'] if v in model.params]
 rows=[]
 for v in keys:
  lo,hi=model.conf_int().loc[v];rows.append({'term':v,'coefficient':model.params[v],'se_cluster_station':model.bse[v],'p_cluster_station':model.pvalues[v],'ci_low':lo,'ci_high':hi})
 pd.DataFrame(rows).to_csv(out/'adjusted_model.csv',index=False)
 model_info={'n_camera_hours':int(model.nobs),'n_station_clusters':int(mdf.sensor_id.nunique()),'r_squared_in_sample':float(model.rsquared),'formula':formula,'warning':'Observational association only; weather, location/time controls do not identify causal traffic effects. Model fit is in-sample.'}
 (out/'model_info.json').write_text(json.dumps(model_info,indent=2))
 # Citywide temporal trend: aggregate first by station-hour, then average stations per local hour.
 v=valid.groupby(['date','hour_of_day'],as_index=False).agg(pm25_mean=('pm25','mean'),traffic_mean=('total_vehicles','mean'),camera_hours=('pm25','size'))
 v.to_csv(out/'hourly_profiles.csv',index=False)
 byhr=valid.groupby('hour_of_day').agg(pm25=('pm25','mean'),traffic=('total_vehicles','mean')).reset_index()
 fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
 axes[0].plot(byhr.hour_of_day,byhr.traffic,marker='o');axes[0].set_ylabel('Vehicles per image (mean)');axes[0].set_title('Camera-hour traffic and attributed PM2.5 across hours')
 axes[1].plot(byhr.hour_of_day,byhr.pm25,marker='o');axes[1].set_ylabel('PM2.5 (OpenAQ reported units)');axes[1].set_xlabel('Local hour (Bangkok)')
 fig.tight_layout();fig.savefig(out/'hourly_patterns.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,5));ax.hexbin(valid.total_vehicles,valid.pm25,gridsize=34,mincnt=1,bins='log');ax.set(xlabel='Mean vehicles per CCTV frame, camera-hour',ylabel='PM2.5 attributed to camera, station-hour',title='Association (density; camera-hour units)');fig.tight_layout();fig.savefig(out/'traffic_pm25_density.png',dpi=160);plt.close(fig)
 print(json.dumps({'audit':audit,'top_correlations':c.head(12).to_dict('records'),'adjusted_log_total':rows[0],'model':model_info},indent=2,default=str))
 return audit,c,rows,model_info

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--csv',default='data.csv');p.add_argument('--out',default='output');a=p.parse_args();run(a.csv,a.out)
