# On-Call Rotation

Engineering teams that own production services take part in the on-call rotation. This page explains how the rotation works, what is expected when you are paged, and how on-call time is compensated.

## Rotation

Each rotation lasts one week, starting Monday at 10:00 CET and ending the following Monday at 10:00 CET. Every rotation has a primary and a secondary engineer. New engineers shadow two full rotations as secondary before they can be primary.

Shift swaps are allowed. Agree the swap with a colleague and update the schedule in PagerTrail at least 48 hours before the shift starts.

## Responding to a page

The primary must acknowledge a page within 15 minutes. If the primary does not acknowledge within 15 minutes, the page escalates to the secondary. If neither acknowledges within 30 minutes, it escalates to the engineering manager of the owning team.

Severity 1 incidents (full outage or data loss) require opening an incident channel in chat and posting a status update every 30 minutes until the incident is resolved.

## After an incident

A written post-mortem is required for every Severity 1 and Severity 2 incident. It must be published within five working days of the incident. Post-mortems are blameless: they focus on the system and process, not on individuals.

## Compensation

On-call engineers receive a flat stipend of 300 EUR per week of primary on-call and 150 EUR per week of secondary on-call. If you are paged between 22:00 and 07:00, you may start work late the next day by the time you spent on the incident.
