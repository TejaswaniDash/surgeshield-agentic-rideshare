---
title: MTA Congestion Relief Zone per-trip charge for High-Volume FHV trips
doc_type: regulation
jurisdiction: NYC / MTA
source_url: https://www.mta.info/fares-tolls/tolls/congestion-relief-zone/about
related_url: https://nyc.gov/assets/tlc/downloads/pdf/industry-notices/industry_notice_24_10_english.pdf
effective_date: 2025-01-05
---
# Congestion Relief Zone (CRZ) charge

- Since January 5, 2025, trips to, from, within or through the Congestion Relief Zone carry a per-trip charge. The zone is Manhattan south of and including 60th Street, with some highways excluded.
- High-volume FHV trips (Uber, Lyft) pay $1.50 per trip in both the peak and overnight periods. Taxis and other FHVs pay $0.75.
- The passenger pays the charge. The platform collects it and remits it to the MTA, so it does not go to the driver or the platform.
- In TLC trip records the charge appears in the `cbd_congestion_fee` column for 2025 data onwards.

## Implications for agents
- Fare estimates in rider messages for CRZ trips must include the $1.50 charge. It must not be shown as surge or as platform revenue.
- Moving drivers into CRZ zones does not by itself trigger a toll on the driver.
- Every SurgeShield Manhattan zone except Lincoln Square East (zone 142) and Upper East Side South (zone 237) lies inside the CRZ. Those two sit partly or wholly north of 60th Street, so the charge applies only when the trip crosses into the zone.
