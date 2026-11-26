# Checks before scientific registration

Both development invocations passed all seven new CPU tests. The first
end-to-end synthetic smoke completed its36ridge fits but failed the second
invocation used to test completed-case resume: newly generated assignment
tuples were compared against JSON-loaded lists. Assignment groups are now
normalized to JSON-compatible lists before comparison and serialization.
This correction was made before registering/freezing this study or fitting
scientific data. The failed attempt remains in development_checks.json and
its synthetic files remain under .cache/statistics_smoke_*.

The corrected synthetic smoke passed36fits, coefficient reloads, parameter
immutability during BN recalibration and completed-study resume. A discarded
four-band GPU model passed all-variable and all-quaternion-component gradient
and update checks. There was no new supervised baseline training.

The native CIFAR rows are explicitly reused from the prior frozen diagnostic,
not refitted under changed preprocessing. BN-only and standardized conditions
have separate records and targets, with check data excluded from calibration
and feature-statistic estimation. All changes use new numerical modules;
predecessor source files and configurations remain unchanged.
