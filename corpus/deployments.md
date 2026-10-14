# Deployment Process

All production changes at Halden Systems ship through the same pipeline. This page describes how releases work, when deployments are not allowed, and how to roll back.

## Pipeline

Every change goes through a pull request with at least one approving review. Changes to payment or authentication code need two approving reviews, one of them from the Security team. Merging to main triggers the CI pipeline, which runs tests and builds a container image.

Deployments use a canary strategy: the new version first receives 5 percent of traffic for 30 minutes. If error rate and latency stay within the service's alert thresholds, the rollout continues to 100 percent automatically.

## Deployment windows

Regular deployments are allowed Monday to Thursday between 09:00 and 16:00 CET. Deployments on Fridays, weekends and public holidays are not allowed, except for hotfixes.

A code freeze applies from 15 December to 5 January. During the freeze only hotfixes for Severity 1 or Severity 2 incidents may be deployed, and each one needs approval from the on-call engineering manager.

## Rollback

If a deployment causes errors, roll back first and investigate afterwards. Use the "Revert release" action in the Shipyard dashboard, which redeploys the previous image within about two minutes. Database migrations must be backwards compatible with the previous release so that a rollback never requires a manual database fix.

## Feature flags

Large features are released behind feature flags managed in Flagpole. A flag that has been fully enabled for 30 days must be removed from the code.
