# Teen social-media time — one item, two subpopulations

A second worked example, deliberately shaped unlike the Pew battery: **one**
3-option **ordinal** item, with marginals reported separately for **two
subpopulations**. Source: "Teen Girls with Less Educated Parents Spend More
Time on Social Media," American Teenagers Survey, 2026 (Survey Center on
American Life; survey of US teens, N=782). The aggregates were transcribed
from the published chart:

| Daily social-media time | College-educated parents | Non-college parents |
|---|---|---|
| Less than one hour | 40% | 35% |
| One to three hours | 37% | 29% |
| Four or more hours | 23% | 35% |

## Structure: one persona bank, two calibration targets

Parent education never enters the personas. There is one battery of teen-girl
response profiles (support generation never sees targets), imported twice with
different `truth` marginals — one file per subpopulation — and the same
measured probability bank is fitted once per group. The comparison of which
personas gain weight under each group's marginals is the substantive result.

```bash
umriss init
umriss battery import --metadata examples/teen_social_2026/teen_social_college.json \
  --battery-id girls_college
umriss battery import --metadata examples/teen_social_2026/teen_social_noncollege.json \
  --battery-id girls_noncollege
umriss battery use girls_college

umriss design import --design examples/teen_social_2026/teen_social_design.yaml --design-id v1
umriss design use v1
umriss design validate

umriss support build --tag teen_social
umriss support export --tag teen_social --model gpt-5.5 --service-name openai
# external execution boundary:
ep run --jobs .umriss/projects/default/runs/teen_social/teen_social.jobs.ep \
  --output .umriss/projects/default/runs/teen_social/teen_social.results.ep
umriss support register-results \
  --results .umriss/projects/default/runs/teen_social/teen_social.results.ep --tag teen_social
umriss support parse --tag teen_social

# the same bank, fitted to each subpopulation's marginals:
umriss fit --support .umriss/projects/default/runs/teen_social/teen_social_probabilities.csv \
  --metadata girls_college --tag fit_college --out fits/college
umriss fit --support .umriss/projects/default/runs/teen_social/teen_social_probabilities.csv \
  --metadata girls_noncollege --tag fit_noncollege --out fits/noncollege
```

## What this example cannot do

Leave-one-out validation needs at least two items with truth marginals; a
one-item battery has nothing to hold out, and `umriss validate marginals`
says so (`battery_too_small`) rather than fitting on nothing. Validation here
must come from an external benchmark — for example, respondent microdata, a
second wave, or the other subpopulation's marginal treated as a transfer test.
