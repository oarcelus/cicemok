# Four-agent delivery workflow

These are standalone role specifications for a deliberate engineering loop. Give
each agent its role file together with the current task, repository access, and
the artifacts named in the file. The roles have intentionally distinct authority:

| Role | Owns | Does not own |
| --- | --- | --- |
| Architect | Discovery, system design, requirements, acceptance criteria, and implementation/test specifications | Production implementation or permanent test code |
| Implementer | Conforming code changes, validation, and resolving confirmed findings | Quietly changing material architecture or product decisions |
| Code Reviewer | Independent code/test audit, finding severity, and closure decisions | Directly fixing production code |
| Tester | Test strategy, test code, test evidence, and defect reproduction | Changing production behavior to accommodate a test |

## Delivery loop

```text
User request
    |
    v
Architect -- detailed, versioned specification --> Implementer
    ^                                             |
    |                                             v
    |<-- material design questions        implementation packet
    |                                             |
    |                                             v
    +---- Reviewer <---- test evidence ----> Tester
                    |                         |
                    +-- findings / requests --+
                              |
                              v
                         Implementer
                              |
                              v
                    Reviewer + Tester verify
```

The Tester and Code Reviewer work as peers: the Tester supplies executable
evidence, while the Reviewer decides whether the evidence adequately addresses
the specification and review risks. Both send actionable feedback to the
Implementer. A requirement or architecture ambiguity returns to the Architect.

## Shared artifacts and identifiers

Use the repository's established locations when they exist. Otherwise, this
convention keeps work traceable:

| Artifact | Suggested location | Minimum contents |
| --- | --- | --- |
| Architecture specification | `docs/specs/<feature>.md` | Context, requirements, design, implementation plan, test plan, risks |
| Implementation packet | Pull request or task record | Requirement traceability, changed files, validation, open assumptions |
| Review report | Pull request or `docs/reviews/<feature>.md` | Verdict, `CR-###` findings, test requests, closure evidence |
| Test report | Pull request or `docs/test-results/<feature>.md` | Coverage map, commands, results, `TST-###` defects, gaps |

Use stable IDs across iterations:

- `REQ-###` for every normative functional or non-functional requirement.
- `AC-###` for individually testable acceptance criteria; each must cite its
  parent `REQ-###`.
- `ADR-###` for material architecture decisions.
- `CR-###` for reviewer findings.
- `TEST-REQ-###` for validation the Reviewer asks the Tester to perform.
- `TST-###` for observed test defects or gaps.

## Specification lifecycle

Every architecture specification names its path, revision, status, and approval
authority in a short header. Use this lifecycle:

`Draft` → `Ready for approval` → `Approved` → `Superseded`.

The user or their designated technical/product decision-maker approves a
revision. Only the named current **Approved** revision may be implemented,
reviewed, or used as the final test oracle; discovery and coverage planning may
use a draft when clearly labeled as provisional. When a material decision
changes, the Architect creates a new revision, marks the old one superseded, and
announces the delta to all downstream roles.

## Quality gate

The change is ready only when the current specification is unambiguous, the
Implementer has supplied evidence for each in-scope requirement, the Reviewer
has no unaccepted blocker/critical findings, and the Tester has either covered
each applicable acceptance criterion or documented a specific accepted gap.
Neither a passing test suite nor an approval alone overrides an unresolved
material specification issue.

## Current repository overlay

For this repository, agents should first recognize that `cicemok/` is a Python
package targeting Python 3.11, the existing `testing/` directory mostly contains
standalone scientific scripts, and several dependencies may require expensive or
external simulation environments. The Architect and Tester should explicitly
distinguish fast, deterministic unit tests from integration checks that need
COMSOL, PyBaMM, model files, or long-running numerical workloads. Do not assume a
test runner, CI command, or external runtime is available without checking.
