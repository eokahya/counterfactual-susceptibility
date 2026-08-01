# Experiment Log

**Status:** No scientific experiments have been completed.

Unit tests, environment inspection, and deterministic formula verification are
Stage 0 engineering checks. They are recorded in `docs/STAGE_0_REPORT.md`, not as
model evidence here.

For every completed scientific run, copy the template below and append it to the
end of this file. Entries are immutable after publication in this log. If an
interpretation changes, append a new linked entry rather than rewriting the old
one.

---

## EXP-YYYYMMDD-NNN — Short descriptive title

- **Status:** completed | failed | aborted
- **Date (UTC):** YYYY-MM-DDTHH:MM:SSZ
- **Authors/operators:**
- **Scientific question or hypothesis:**
- **Prerequisite experiment IDs:** none
- **Planned versus exploratory:**
- **Code commit:** full commit hash
- **Dirty-tree status:** clean | dirty, with diff/artifact reference
- **Configuration:** tracked path plus an archived resolved configuration
- **Random seeds:**
- **Upstream package:** repository URL, exact commit, package version
- **Model:** identifier and immutable revision
- **Transcoder/CLT:** identifier and immutable revision
- **Prompt inputs:** identifiers, tracked file, or deterministic generation recipe
- **Behavior metric:** name, direction, token/position convention, target and
  contrast where applicable
- **Hardware:** accelerator, memory, CPU, RAM
- **Software:** OS, Python, PyTorch, accelerator/runtime, relevant dependencies
- **Intervention:** source/target selection and exact `alpha` convention
- **Primary metrics and results:** signed values and uncertainty where applicable
- **Replacement-model result:** not run | summary and artifact
- **Underlying-model result:** not run | summary and artifact
- **Failures, warnings, and anomalies:**
- **Peak memory and wall-clock time:**
- **Artifacts:** paths to raw metadata, metric tables, logs, figures, and checksums
- **Deviation from plan:** none | description and rationale
- **Decision:** continue | revise | stop
- **Follow-up:**

---

Never record a planned, configured, or partially scaffolded run as completed.
