---
title: New York price-gouging law (GBL 396-r) and the 2014 Uber emergency surge agreement
doc_type: regulation
jurisdiction: NY State
source_url: https://www.nysenate.gov/legislation/laws/GBS/396-R
related_url: https://phys.org/news/2014-07-ny-price-limit-car-uber.html
effective_date: 2014-07-08
---
# Price gouging during abnormal market disruptions

## The statute
New York General Business Law section 396-r prohibits charging unconscionably excessive prices for goods and services vital to the health, safety or welfare of consumers during an "abnormal disruption of the market". That term covers weather emergencies and natural disasters, power failures, civil disorder, war and similar events, and it applies especially when a state of emergency is declared.

## The 2014 agreement with Uber
In July 2014 the New York Attorney General announced an agreement with Uber on dynamic pricing during such disruptions. It followed public criticism of very high fares during severe weather. Under the agreement:
- Surge multipliers during an abnormal disruption are capped at the normal range charged in the preceding 60 days, excluding the three highest multiples.
- It covered Uber's dynamically priced products in New York, and Uber said it would apply a similar national policy.

## Operational rule derived for agents (SurgeShield policy P-1)
- If a disruption is flagged as a declared emergency or an abnormal market disruption (severe weather emergency, power outage, transit emergency declared by authorities), the Pricing agent must compute the emergency cap as follows. Take the 60-day history of multipliers for the zone, drop the three highest values, and use the maximum of what remains. The cap is that value or the configured fallback, whichever is lower.
- Ordinary demand spikes (a concert ending, a normal rainy evening) are not abnormal disruptions, so the normal surge cap applies.
- Whenever the classification is uncertain, the Compliance agent must route the plan to human approval.
