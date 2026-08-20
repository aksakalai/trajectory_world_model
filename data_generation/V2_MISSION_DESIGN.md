# V2 Mission Design

## Status

The catalog, construction-time certification, runtime playback, immutable
planner, schemas, full-plan verifier, validators, bounded pre-recording
resolver, 85-video representative review tooling, and optional 186-video
exhaustive V2 review tooling described here are implemented.

The approved 2,222,222-frame V2 campaign contains 1,555,555 semi-Markov frames
and 666,667 prescribed-mission frames. Its original certification passed all
4,940 prescribed mission recipes. The complete final resolved plan passed
5,459/5,459 recipes with zero rejects and is bound to the Windows Unreal Engine
5.8 executable/package used for certification. RGB production capture has not
yet run.

## Purpose

V2 missions provide rare, visually understandable grenade interactions that
persistent semi-Markov play may not produce often enough. They complement
random play; they do not replace it and do not manufacture physics outcomes.

The final frame mixture is:

- 70% persistent semi-Markov play;
- 30% prescribed V2 missions.

Semi-Markov play covers ordinary misses, floor impacts, natural bounces, rolls,
settling, varied controls, multi-throw play, and unexpected outcomes. Missions
cover deliberate object surfaces, edges, apexes, rims, ramp geometry, wall
corners, and arena exits.

## Non-negotiable rules

1. Every grenade uses the canonical game launch speed, cooldown, and immutable
   `FGrenadeSimConfig`.
2. Preview and realized motion use the same launch state and physics config.
3. A recipe may vary only legitimate player-controlled initial conditions and
   actions: spawn, camera yaw/pitch, action timing, and post-throw attention.
4. Mission solutions are constructed and certified before video generation.
5. The planner and verifier never mutate recipes. A separate pre-recording
   resolver may replace a rejected quota slot only through the bounded,
   same-type policy below; the recorder never searches or substitutes.
6. A technical retry may replay the identical immutable recipe after a machine
   interruption; it may not change the recipe or seed.
7. Missions guarantee only their named interaction and its visibility.
   Subsequent bounce count, roll, rest, and final position remain natural.
8. No privileged trajectory points or solver geometry become learning targets.

## Catalog: 62 meaningful types

Each type receives the nearest whole-frame allocation to `3/620`
(approximately 0.4839%) of total dataset frames. At most one frame separates
type targets. Variations within a type are continuous samples, not additional
semantic labels.

### 1. Broad object-surface impacts - 13 types, 6.5%

Rectangle:

1. north vertical face;
2. south vertical face;
3. east vertical face;
4. west vertical face;
5. top surface away from its perimeter.

Pyramid:

6. north sloping face;
7. south sloping face;
8. east sloping face;
9. west sloping face.

Sphere:

10. approach from the north quadrant;
11. approach from the south quadrant;
12. approach from the east quadrant;
13. approach from the west quadrant.

Surface impacts must land safely inside their intended face/region and away
from an edge corridor. Contact height, distance, arc, and obliqueness vary.

### 2. Object edge and apex impacts - 13 types, 6.5%

Rectangle vertical corner edges:

1. north-east;
2. north-west;
3. south-east;
4. south-west.

Rectangle upper perimeter edges:

5. north;
6. south;
7. east;
8. west.

Pyramid:

9. north-east sloping ridge;
10. north-west sloping ridge;
11. south-east sloping ridge;
12. south-west sloping ridge;
13. apex region.

An edge or apex is a narrow physical contact corridor based on collision
geometry and grenade radius, not a zero-width mathematical line or point. Each
type must support direct and oblique approaches that visibly produce different
natural deflections.

### 3. Wall and corner rebounds - 12 types, 6%

Direct wall contacts:

1. north wall;
2. south wall;
3. east wall;
4. west wall.

Oblique wall contacts:

5. north wall;
6. south wall;
7. east wall;
8. west wall.

Two-wall corner interactions:

9. north-east corner;
10. north-west corner;
11. south-east corner;
12. south-west corner.

Oblique direction and, for corners, first-contact order vary across repetitions
without becoming separate labels.

### 4. Hoop interactions - 10 types, 5%

Clean passages:

1. negative-X to positive-X;
2. positive-X to negative-X.

Rim contacts from each direction:

3. upper rim, negative-X to positive-X;
4. lower rim, negative-X to positive-X;
5. left rim, negative-X to positive-X;
6. right rim, negative-X to positive-X;
7. upper rim, positive-X to negative-X;
8. lower rim, positive-X to negative-X;
9. left rim, positive-X to negative-X;
10. right rim, positive-X to negative-X.

Clean passages vary across safe center and offset regions. Rim contacts target
the physical tube body with margin; they are not tangent near-misses.

### 5. Ramp interactions - 8 types, 4%

1. uphill surface impact;
2. downhill surface impact;
3. lateral crossover, left to right;
4. lateral crossover, right to left;
5. left side-edge impact;
6. right side-edge impact;
7. high-end lip impact;
8. low-end lip impact.

Lateral crossover launch and landing regions must lie on opposite sides of the
ramp, and the simulated path must geometrically cross the ramp body. Merely
moving laterally in front of or behind the ramp is not a crossover.

### 6. Deliberate out-of-bounds - 4 types, 2%

1. north boundary exit;
2. south boundary exit;
3. east boundary exit;
4. west boundary exit.

Directness, obliqueness, and exit height vary. The chosen boundary and useful
arena context remain visible. Natural out-of-bounds throws also remain legal in
semi-Markov play.

### 7. Trajectory-control demonstrations - 2 types, approximately 0.9677%

1. manual trajectory-preview toggle cycle;
2. throw, cooldown-hidden preview, reload, and unchanged-angle preview reopen.

These demonstrate the global Q/E state machine. They do not alter grenade
physics or create additional trajectory behavior modes.

## Constructing a mission solution

### Controllable variables

The builder may choose:

- a valid player spawn in the arena;
- camera yaw and pitch reachable through ordinary inputs;
- a target point or contact corridor on the named geometry;
- action dwell durations;
- post-release camera and movement actions.

It may not change grenade speed or any simulation parameter.

### Ballistic selection

For each type, the builder selects a stratified desired interaction point and
solves the canonical throw from a valid spawn. The low ballistic branch is the
default. The former practice of selecting the upper mathematical root merely to
obtain a hit is prohibited.

A higher-looking throw is allowed only when it remains human-readable and keeps
arena context visible. If an interaction can be reached only through a
sky-only, near-vertical aim, that spawn/target pairing is not a usable solution.
The builder must relocate the player or choose another point inside the same
semantic region.

The exact resulting launch is simulated with the same function and config used
by preview and realized play. A construction assertion confirms the named
contact, crossing, passage, or exit. Inside the solution builder, failure aborts
construction; the builder does not silently select another production seed.

### Certified variation region

Every type owns a bounded, known-working solution region. Repetitions first
progress through deterministic discrete coverage cells and then refine those
cells with seeded continuous offsets. The cell identity is stored independently
from the exact recipe identity, so a larger frame budget extends the same
coverage prefix instead of starting a new random schedule. Legitimate dimensions
include:

- launch distance;
- position along a surface, edge, rim, or boundary;
- contact height or offset;
- lower and moderate arcs where feasible;
- direct and oblique approach;
- initial composition and target placement in the image;
- post-throw attention style.

The first repetition uses a stable central solution. Later repetitions spread
progressively across the certified region instead of clustering randomly.
Two-wall missions explicitly cover both valid first/second contact orders. Wall
targets remain on their physical wall plane; sphere and hoop targets remain on
their physical surface or rim centerline. Seeds determine deterministic safe
mutations but never determine whether a mission works.

## Camera and action railguards

Only two visual railguards are required.

### 1. The mission region stays visible

The intended surface, edge, apex, rim, ramp region, wall/corner, or arena
boundary must be visible from the opening frame through the required
interaction and final mission frame.

While Q is held, the useful trajectory-preview segment and the intended region
must be visible together. The full infinite or long-range path need not fit in
the image; the relationship between the visible arc and target must be clear.

### 2. No sky-only or floor-only opening

The opening must contain recognizable arena and mission context. It may not
start with effectively the entire image occupied by empty sky or ground.
Target-centered, eye-level-biased camera construction should satisfy this
without a pixel-percentage validator.

These are construction rules, not dataset distribution thresholds. Framing may
still place the target left, right, high, low, near, or far inside the image.

### Human-readable timing

A typical mission has:

1. 0.75-1.5 seconds establishing the target and surrounding scene;
2. 0.6-1.2 seconds of visible Q-held trajectory preview;
3. gradual aim adjustment when adjustment is used;
4. one canonical throw after Q was visible in the preceding observation;
5. camera behavior that preserves the named interaction and initial deflection;
6. a short natural aftermath without waiting for prescribed settling.

One- or two-frame semantic gestures, aim jitter, and instantaneous left-right or
up-down corrections are not valid mission behavior. Semi-Markov play remains
free to produce rare short or contradictory controls.

## Immutable planning and frame allocation

The planner first emits one mandatory recipe for every one of the 62 types. It
interleaves families and spatial regions so early partial runs are not localized
to one arena area.

After the mandatory pass, it selects the type with the largest credited-frame
deficit relative to its whole-frame `3/620` target. Selection is frame-based,
not episode-count based, so longer flights do not receive accidental excess
weight.

At a 700,000-frame budget:

- semi-Markov receives 490,000 frames;
- all missions receive 210,000 frames;
- each mission type receives approximately 3,387 frames.

### Minimum feasible budget

Mission duration is known from its frozen action sequence, certified event time,
and fixed aftermath. The planner calculates the minimum budget rather than
copying any V1 or historical V2 value:

```text
minimum_total_budget = max(
    semi_markov_mandatory_frames / 0.70,
    mandatory_frames_for_family / family_frame_share
    for every mission family
)
```

The planner refuses a production budget below this floor. Every feasible plan
contains all 62 mission types at least once. No runtime recipe is invented to
repair a plan.

With the default 150-second semi-Markov duration and the certified 20 Hz
observation rate, the calculated floor is 33,274 frames. The planner derives and
recomputes this value from mandatory durations and shares rather than storing it
as a legacy constant. Observation rates other than 20 Hz are rejected until they
have their own physics and camera certification.

Train/evaluation assignment uses deterministic, disjoint recipe identities and
seeds. When the budget permits, every type is represented in both splits.

## Full-plan construction verification

Static planning proves allocation, schema, identity, and catalog consistency;
it does not prove that every generated launch remains physically feasible in
the current arena build. Before recording, `Scripts/certify_v2_plan.py` must
therefore certify the entire immutable plan in one Unreal session.

The verifier loads the arena once and runs the canonical construction simulator
for every mission recipe without capturing images, encoding WebP, writing
Parquet, or crediting frames. Semi-Markov recipes are reported as construction
not applicable. The bound report records every recipe result and fingerprints:

- the collection plan and complete recipes ledger;
- the sealed assignment set;
- the generator source;
- the exact executable and packaged runtime;
- the canonical physics identity and Unreal version.

Any rejected recipe makes the command exit unsuccessfully and blocks recording
of that candidate plan. The worker repeats construction certification
immediately before each mission as a second fail-closed check against runtime
mismatch.

The verifier is not a candidate search. It never changes a seed, selects a
replacement, or approves a subset. Its only valid production result is a
complete report with every recipe certified.

### Bounded pre-recording resolution

`Scripts/resolve_plan_certification.py` consumes the complete verifier report;
it does not modify the verifier or its acceptance thresholds. For each rejected
quota slot it:

1. preserves the original recipe and engine failure evidence;
2. tries up to ten distinct candidates of the same mission type and scenario;
3. orders attempts from diverse to increasingly conservative variation;
4. if those fail, may mutate toward an already certified recipe of the same
   mission type while retaining the failed slot's type, scenario, split, and
   frame cap;
5. records every attempt and its certification result;
6. emits a new immutable resolved collection; and
7. runs `certify_v2_plan.py` over the complete resolved collection to create the
   ordinary plan/build-bound recording certificate.

An unresolved slot keeps the campaign incomplete. No partial plan is approved.
The approved V2 campaign needed no replacements: 4,940/4,940 original mission
recipes certified.

## Crediting, failures, and validation

Semi-Markov episodes are always credited when technically valid. No behavioral
or statistical threshold can reject them.

Mission recipes are not part of a recording-time accept/reject search. Their
named event is an invariant already established during construction and final
plan certification. If runtime output disagrees, generation stops as a code or
physics regression. It does not credit a replacement seed.

`planned_credited_frames` is allocation metadata, not a storage truncation
instruction. Finalization retains every valid observation and transition in the
complete episode. The worker reports both produced and credited observations;
training should use complete episodes and must never join different episode IDs.

Machine checks cover:

- immutable recipe identity and complete frame/transition alignment;
- equality of requested and completed episode counts (empty or partial
  "complete" shards are invalid);
- sealed plan/assignment/replay identity through worker output and review media;
- action and Q/E semantics;
- canonical physics configuration identity;
- preview-versus-realized parity;
- named mission interaction evidence, including physical position, normal,
  incoming velocity, direction, clearance, and ordered contacts where relevant;
- the two camera construction railguards;
- image and Parquet integrity.

Machine checks do not approve visual quality. Human review decides whether
pacing, framing, variation, and interactions are understandable.

## Human review sets

The current representative contract is the fixed 85-video V1/V2 suite: 19 V1
videos and 66 V2 videos. It was generated and accepted during the current
project review. This media audit is separate from plan certification and
production data capture.

The repository also supports an optional exhaustive V2 review: exactly three
384x384 examples for every type, or 186 videos total.

The three examples sample separated parts of the certified region:

1. a central/direct solution;
2. an oblique or offset solution from one side;
3. a separated distance, arc, or opposite-side solution.

For edge and apex types, the examples must show different natural deflections
while contacting the same intended corridor. Review covers validity, visibility,
human-readable timing, and repetition. A bad example requires fixing the mission
definition or certified region globally; it is never replaced by a friendlier
seed.

Video review checks usefulness and readability; construction certification
checks physics and named events. Rendering a video is not a second physics
verifier.

## Implementation status

Implemented and verified:

1. catalog-count, share, deterministic-identity, and fail-closed recorder tests;
2. focused geometry, timing, camera, variation, and canonical-identity tests for
   all 62 types;
3. the construction-time low-branch ballistic and camera solution builder;
4. frozen deterministic solution regions and immutable mission recipes;
5. exact whole-frame 70/30 allocation, mandatory pass, deficit scheduling, and
   calculated feasibility floor;
6. runtime playback using ordinary player inputs and canonical throws;
7. technical and invariant validation without statistical acceptance gates;
8. fixed 85-video representative tooling and optional 186-video V2 tooling;
9. one-session full-plan construction certification with immutable build and
   plan bindings;
10. bounded pre-recording replacement lineage and full resolved-plan
    recertification;
11. the approved 2,222,222-frame V2 campaign, passing 4,940/4,940 original
    mission recipes and 5,459/5,459 final total recipes.

Remaining operational work is RGB capture, exact produced-frame inventory, and
Linux-specific build binding/recertification if capture moves from the currently
certified Windows build. Movement V1 mission semantics and the shared V1/V2
semi-Markov policy remain protected by regression tests.
