---
title: Postmortem - JFK delayed arrival bank (sample, synthetic)
doc_type: postmortem
incident_kind: airport_bank
zones: [132, 138]
---
# Postmortem: JFK delayed arrival bank (sample, synthetic)

**Summary.** Weather delays compressed arrivals into a 20:00-22:00 bank, and JFK requests reached 2.7x. Waits in the app-pickup area peaked at 14 minutes.

**What worked.** Notifying drivers 35 minutes ahead using the flight-delay signal. Telling them to follow airport staging instructions prevented curb-waiting complaints.

**What did not.** LaGuardia was over-incentivized while its own demand was normal. That cost $900 of waste.

**Lesson.** Size airport incentives with the forecast gap, not with demand alone. Do not mirror JFK actions to LGA by default.
