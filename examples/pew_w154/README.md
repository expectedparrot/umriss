# Pew ATP Wave 154 tutorial data

This example uses the `DIFF1` battery from Pew Research Center's nationally
representative American Trends Panel Wave 154, fielded among U.S. adults in
September 2024.

The source microdata contains 6,104 normalized respondent records and a survey
weight. It is used only to derive the five weighted marginal distributions in
`weighted_marginals.csv` and `pew_w154_metadata.json`. Respondent rows,
covariates, and respondent-level answer combinations are not passed to model
prompts, calibration, or leave-one-out evaluation.

Each leave-one-out fold fits support weights against four aggregate marginals
and predicts the omitted fifth aggregate marginal.

The 12 support prompts were executed through Expected Parrot under Results UUID
`c78751ef-75e2-4184-95ff-3d1df6c7ac18`.
