---
title: Marketplace Disruption Playbook (sample internal policy)
doc_type: internal_policy
owner: Marketplace Ops (sample - adapt to your organization)
version: 3.2
---
# Marketplace Disruption Playbook

This is a sample internal policy written for the SurgeShield reference project. It is not any real company's policy.

## P-1 Pricing
- Normal surge cap: 2.5x.
- Abnormal market disruptions are governed by the NY price-gouging rule summarized in `ny_price_gouging_gbl_396r.md`. Apply the 60-day emergency cap.
- Surge must not be raised in a zone solely because riders there have fewer alternatives (for example, during a transit outage with no other transport). Transit outages count as abnormal disruptions when an authority has declared service suspended.

## P-2 Driver incentives and budget
- Incentives are relocation bonuses and per-trip boosts, paid on top of TLC minimum pay.
- Total incentive commitment per incident (per planning cycle) up to $500 can be auto-approved. From $500 to $5,000 needs a human Marketplace Ops approver. Anything over $5,000 is rejected; split the incident or escalate to leadership. (These limits are sized for the 20-zone reference twin; scale them to your market.)
- Incentives must be offered to every eligible driver within the target radius. No individual targeting based on protected characteristics or past acceptance behavior.

## P-3 Communications
- Send drivers no more than two disruption messages per hour.
- Never encourage speeding, unsafe stopping, or waiting on airport curbs or bus lanes.
- Never promise guaranteed earnings except for the computed, committed bonus amount.
- Rider messages must show the estimated wait time and that prices are higher, plus a lower-cost alternative (wait-and-save, shared ride, or transit when it is running).

## P-4 Accessibility
- Maintain WAV coverage (see `tlc_accessibility_wav.md`).

## P-5 Human-in-the-loop and auditability
- Every plan must carry a rationale, the model versions used and the retrieved policy citations.
- If an approver does not respond within 10 minutes, a plan inside auto-approve limits proceeds. Any other plan escalates to the on-call lead and the workflow holds.
- If KPIs have not recovered after 2 hours, escalate.

## P-6 Exit criteria
- An incident is resolved when, for one monitoring interval, the median rider wait in the affected zones is below 6 minutes and at least 90% of requests are fulfilled. Unwind all incentives on resolution.
