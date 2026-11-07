from pathlib import Path
import json
import pandas as pd
from scipy.stats import spearmanr
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
P=Path(__file__).parent;O=P/'output'
a=json.loads((O/'audit.json').read_text()); m=json.loads((O/'model_info.json').read_text());c=pd.read_csv(O/'camera_hour_spearman.csv');adj=pd.read_csv(O/'adjusted_model.csv');d=pd.read_csv(O/'camera_hours.csv');v=d.dropna(subset=['pm25'])
subsets={'All matched camera-hours':v,'Exclude zero PM2.5 readings':v[v.pm25>0],'Nearest reporting station only':v[v.is_fallback==0],'Station within 1 km':v[v.station_distance_m<=1000]}
rows=[]
for name,df in subsets.items():
 rho,p=spearmanr(df.total_vehicles,df.pm25);rows.append({'sample':name,'n_camera_hours':len(df),'rho_traffic_pm25':rho,'p_naive':p})
s=pd.DataFrame(rows);s.to_csv(O/'sensitivity_spearman.csv',index=False)
D=Document();sec=D.sections[0];sec.header.paragraphs[0].text='DSDE PROJECT  /  COMPONENT 2  /  REVISED ANALYSIS'
normal=D.styles['Normal'];normal.font.name='Aptos';normal.font.size=Pt(9)
normal.paragraph_format.space_after=Pt(6)
D.add_heading('Component 2 — Data Science',0)
D.add_paragraph('Corrected PM2.5 analysis of Bangkok CCTV traffic observations',style='Subtitle')
D.add_paragraph('Analysis period: 25–31 August 2026  |  Bangkok CCTV observations')
D.add_heading('Executive summary',level=1)
D.add_paragraph('The submitted CSV contains actual populated PM2.5 values and station identifiers alongside CCTV vehicle counts, weather and place variables. Unlike the earlier analysis that treated PM2.5 as unavailable, this revision tests traffic and measured ambient PM2.5 at a camera-hour level. The relationship is small and negative in raw ranks; its adjusted estimate does not meet a two-sided 5% significance threshold. These findings are associational and limited by shared station observations and spatial mismatch.')
D.add_heading('1. Data and analytical approach',level=1)
D.add_paragraph('The dataset combines vehicle counts from CCTV images with ambient PM2.5 station readings, weather, and location attributes. Vehicle counts are the main explanatory variables (X), and PM2.5 is the outcome (Y). Analyses use one camera-hour observation to avoid treating repeated frames sharing the same PM2.5 reading as independent observations.')
D.add_paragraph('The PM2.5 values in this analysis are the values supplied in the prepared dataset. Results are descriptive and associational; data-source attribution and potential differences between station and roadside conditions are considered in the limitations.')
D.add_heading('2. Dataset and quality audit',level=1)
for k,txt in [('image_rows','Image rows'),('camera_hours','Distinct camera-hours'),('cameras','Cameras'),('stations','Distinct PM2.5 sensors'),('pm25_valid_images','Image rows with PM2.5'),('pm25_valid_camera_hours','Camera-hours with PM2.5'),('zero_pm25_camera_hours','Camera-hours with zero PM2.5'),('pm25_conflicts_within_camera_hour','Conflicting PM2.5 values within camera-hours')]:
 D.add_paragraph(f'{txt}: {a[k]:,}',style='List Bullet')
D.add_paragraph(f"Observation window: {a['start']} to {a['end']} (local time). {a['fallback_share_pm25_camera_hours']:.1%} of attributed camera-hour readings use a fallback station, and the maximum matched sensor distance is {a['max_station_distance_m']:,.0f} metres. Zero readings may be valid or may reflect sensor-level issues; the origin must be audited separately.")
D.add_paragraph('PM2.5 is measured per station-hour but reused across nearby cameras. Camera-hour aggregation removes multiple image copies; it does not create an independent pollution observation for every camera-hour. Naive small p-values can therefore overstate evidence.')
D.add_heading('3. Correlations and relationships',level=1)
D.add_paragraph('Spearman rank correlation across camera-hour averages. The p and Benjamini–Hochberg q values in the accompanying CSV do not correct for dependence among cameras sharing an OpenAQ station; treat them as exploratory rankings, not causal or confirmatory tests.')
T=D.add_table(rows=1,cols=4);T.style='Light Shading Accent 1'
for cell,text in zip(T.rows[0].cells,['Variable','Spearman ρ','Pairs','BH q (naive)']):cell.text=text
for _,r in c.iterrows():
 z=T.add_row().cells
 for cell,value in zip(z,[r['variable'],f"{r.spearman_rho:+.3f}",str(int(r.n_camera_hours)),f"{r.q_bh:.2g}"]):cell.text=value
D.add_paragraph('The strongest raw association among the tested measurements is wind speed (ρ = −0.277). Total traffic is weakly negatively associated with attributed PM2.5 (ρ = −0.101). This should not be interpreted as traffic reducing pollution; shared time-of-day, weather and background pollution can explain opposite-signed raw patterns.')
D.add_heading('4. Conditional statistical model',level=1)
r=adj[adj.term=='log_total'].iloc[0]
D.add_paragraph('OLS: PM2.5 ~ log(1 + mean vehicles/image) + camera fixed effects + local-hour fixed effects + calendar-date fixed effects + temperature, precipitation, wind speed, relative humidity, cloud cover, and surface pressure. Standard errors are clustered by PM2.5 sensor (31 clusters).')
D.add_paragraph(f"Estimated coefficient for log(1 + traffic): {r.coefficient:+.3f}; 95% station-clustered CI [{r.ci_low:+.3f}, {r.ci_high:+.3f}]; p = {r.p_cluster_station:.3f}. Model n = {m['n_camera_hours']:,} camera-hours; in-sample R² = {m['r_squared_in_sample']:.3f} (not a forecasting score).")
D.add_paragraph('Conclusion: the adjusted total-traffic association is not significant at α = 0.05. The interval includes small negative and near-zero positive values. This is not evidence that vehicle emissions are harmless, nor evidence against all longer-lag traffic effects.')
D.add_heading('5. Sensitivity of the raw correlation',level=1)
t=D.add_table(rows=1,cols=3);t.style='Light Shading Accent 1'
for cell,text in zip(t.rows[0].cells,['Sample','Camera-hours','Spearman ρ']):cell.text=text
for _,row in s.iterrows():
 for cell,value in zip(t.add_row().cells,[row['sample'],f"{int(row.n_camera_hours):,}",f"{row.rho_traffic_pm25:+.3f}"]):cell.text=value
D.add_paragraph('These are descriptive subset checks only. Changes in sample composition, station identity, and spatial coverage can move raw correlations. The zero-value and fallback filters must not be presented as proof of measurement validity.')
D.add_heading('6. Patterns over time',level=1)
D.add_picture(str(O/'hourly_patterns.png'),width=Inches(6.3))
D.add_paragraph('Traffic counts are instantaneous images taken periodically, not continuous vehicle flows; averaging frames is a proxy for visible traffic occupancy. PM2.5 values are station-hour concentrations attributed to each camera. Their hourly profiles cannot establish whether emissions preceded a rise in air pollution.')
D.add_heading('7. Distribution of traffic and PM2.5',level=1)
D.add_picture(str(O/'traffic_pm25_density.png'),width=Inches(6.3))
D.add_heading('8. Findings and implications',level=1)
for text in ['A weak negative rank association between traffic occupancy and ambient PM2.5 appears in the supplied data; it is not evidence of a protective traffic effect.','Wind speed has a stronger negative raw association with PM2.5 than traffic does; meteorological context is essential to interpretation.','Attribution depends on 31 PM2.5 sensors for 100 cameras, and approximately 40% of matched camera-hours use a fallback station. Near-camera pollution is not directly measured.','A controlled log-traffic model has an interval crossing zero with station-clustered uncertainty, so a reliable independent traffic effect is not established with this seven-day observational sample.','The next useful analysis is a station-hour quality check, lagged-traffic comparison, and out-of-time validation; the current sample does not support causal inference.']:
 D.add_paragraph(text,style='List Bullet')
D.add_heading('9. Dashboard handoff for Component 3',level=1)
D.add_paragraph('Use the ready-to-load CSV files under output/dashboard/. Each table has an explicit observation level so maps, time-series plots and correlation panels can be built without re-aggregating image rows.')
for text in [
 'cameras.csv — one row per camera; latitude/longitude for map markers, average traffic and PM2.5, PM2.5 coverage, and source-distance indicators.',
 'camera_hours.csv — one row per camera-hour; traffic counts by vehicle class, PM2.5, weather, station provenance, date and local hour. Use for linked filters and time-series charts.',
 'camera_hourly_profiles.csv — one row per camera and hour-of-day; supports per-camera diurnal charts.',
 'station_hours.csv — one row per station and hour; use for sensor-level plots and to understand repeated pollution values across cameras.',
 'correlations.csv — variable-wise Spearman correlation table with sample sizes and exploratory multiple-testing-adjusted q values.']:
 D.add_paragraph(text,style='List Bullet')
D.add_paragraph('Recommended dashboard panels: camera map with a pollutant/traffic selector; hourly traffic and PM2.5 trend charts; vehicle-class filters; correlation ranking; and a data-quality view showing PM2.5 coverage, station distance, and fallback attribution. Display units and observation level clearly. Correlations must not be labelled as causal effects.')
D.add_heading('10. Reproducibility and study limitations',level=1)
D.add_paragraph('The accompanying analyze_pm25.py generates the audit, camera-hour summaries, Spearman correlations, adjusted regression, sensitivity summaries, figures and dashboard CSVs. Running make_report.py after the analysis reproduces this document. Input: data/processed/data.csv; generated results: Component2_DataSci/output/.')
D.add_paragraph('Limitations: approximately seven days of data; sensor-hour PM2.5 values shared across cameras; CCTV image counts are an occupancy proxy rather than measured traffic flow; potential detector errors; uncertain validity of zero PM2.5 readings; station-to-camera distance and fallback attribution; weather and time confounding; and no causal identification or out-of-time predictive validation.')
D.save(P/'Component2_Findings_REVISED.docx')
print('saved',P/'Component2_Findings_REVISED.docx')
