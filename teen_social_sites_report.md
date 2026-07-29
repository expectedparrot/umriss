# Social-media sites reported by the weighted teen-girl agent list

The five personas answered one checkbox question through GPT-5.5:

> Which of the following social media sites or platforms do you spend time on?
> Select all that apply.

Each persona was interviewed once. The answers were then aggregated with each
of the two fitted population weight vectors. The weights were not shown to the
model.

| Platform | College-educated parents | Non-college parents |
|---|---:|---:|
| YouTube | 100.0% | 100.0% |
| Reddit | 100.0% | 100.0% |
| Instagram | 72.4% | 75.3% |
| TikTok | 47.9% | 54.4% |
| Facebook | 47.9% | 54.4% |
| X (formerly Twitter) | 47.9% | 54.4% |
| Discord | 47.9% | 54.4% |
| Pinterest | 34.2% | 35.6% |
| Snapchat | 23.9% | 39.6% |
| Threads | 0.0% | 0.0% |
| Tumblr | 0.0% | 0.0% |
| Another site or platform | 0.0% | 0.0% |
| None of these | 0.0% | 0.0% |

The weighted mean number of selected platforms was 5.22 for the
college-parent calibration and 5.68 for the non-college-parent calibration.

## Interpretation

The non-college-parent calibration assigns more mass to the high-use personas,
so it produces higher estimated platform reach, especially for Snapchat
(39.6% versus 23.9%). YouTube and Reddit were selected by every persona and
therefore have 100% estimated reach under both weighting schemes.

These are model-generated answers from only five support personas, not
observed survey estimates. A platform share is a weighted aggregation of one
resolved checkbox response per persona. The results should be treated as a
demonstration of downstream AgentList use, not as population measurement.

## Run record

- Model: GPT-5.5
- Interviews/model calls: 5
- Valid results: 5
- Actual inference cost: $0.015545
- Results package: `teen_social_sites.results.ep`
- Extracted answers: `teen_social_sites_answers.csv`
- College weights: `fits/college/fit_college_weights.csv`
- Non-college weights: `fits/noncollege/fit_noncollege_weights.csv`
