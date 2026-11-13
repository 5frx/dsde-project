# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Streamlit (Python), Plotly and pydeck charts. Run with `cd INSIGHTS && streamlit run app.py`. Separate from the existing VIZ dashboard, which must not be modified.

## Users

A university data science team (2190513, Chulalongkorn ICE) presenting a course project to graders, in a live demo and a recorded 15-minute video. Secondary: classmates browsing the Discord project showroom.

## Product Purpose

Turn one week of Bangkok traffic CCTV (100 BMA cameras, 25–31 Aug 2026, 06:00–21:00, a frame every 10 minutes) into findings a viewer can understand and trust: when and where traffic happens, what weather and air quality do alongside it, and where the measurements themselves are weak. Success: a grader leaves with five or six concrete findings and believes each one because the evidence sits beside it.

## Positioning

Built on vehicle counts the team extracted themselves with YOLOv8 from raw city cameras, joined to official air-quality and weather records and OpenStreetMap places. It reports its own blind spots (dusk detection loss, low-cost PM2.5 sensors) instead of hiding them.

## Operating Context

Shown full-screen on a laptop or projector during a presentation, and screen-recorded. Viewers read it in sequence, not as a daily tool.

## Capabilities and Constraints

- Data: Component2_DataSci/data/processed/data.csv (team pipeline) plus INSIGHTS/data (scraped: Nominatim districts, OSM rail stations, Open-Meteo sunset; reused: Air4Thai hourly PM2.5 for 79 stations, METAR from Don Mueang and Suvarnabhumi, OSM road classes, frame brightness).
- One week only; associations, not causes. No truck-ban claims (rule hours could not be verified).
- Camera frames: a set of real CCTV stills is available in INSIGHTS/assets/frames.

## Evidence on Hand

All figures come from INSIGHTS/data/derived (built by build_insights.py). No testimonials or external validation exist; do not invent any.

## Product Principles

- Every finding is a sentence first, then the chart that proves it.
- Show uncertainty and data caveats next to the claim, not in a footnote.
- Real camera images over decoration.
