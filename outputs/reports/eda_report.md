# Exploratory Data Analysis: BankSim Development Sample

## Scope and prevalence

This analysis uses only `data\processed\banksim_sample.csv`. It does not read the original source dataset.

The sample contains **50,000 rows** and **10 original columns**, with **7,200 fraud** and **42,800 legitimate** transactions. Its fraud prevalence is **14.400000%** (expected sample baseline: 14.400000%), compared with **1.210811%** in the original dataset.

Because this sample retains every fraud transaction while downsampling legitimate transactions, its prevalence and group fraud rates are descriptive of this enriched sample, not unbiased estimates of the original population.

## Data quality and structure

- Shape: 50,000 rows × 10 columns.
- Exact duplicate rows: 0.
- Step coverage: 180 observed values, from 0 to 179.
- Fraud rate: 14.400000%.
- Missing values by column:

No missing values were found.

## Transaction amounts by fraud label

| fraud_label | count | mean | median | std | min | max |
| --- | --- | --- | --- | --- | --- | --- |
| Legitimate (0) | 42800 | 32.061 | 26.700 | 34.333 | 0.000 | 1805.590 |
| Fraud (1) | 7200 | 530.927 | 319.175 | 835.587 | 0.030 | 8329.960 |

Figure: `../figures/amount_by_fraud.png`. The box plot uses log(1 + amount) to improve readability and omits plotted outlier markers; summary statistics use the original amounts.

## Category patterns

| category_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 'es_leisure' | 476 | 474 | 99.580% |
| 'es_travel' | 596 | 578 | 96.980% |
| 'es_sportsandtoys' | 2143 | 1982 | 92.487% |
| 'es_hotelservices' | 641 | 548 | 85.491% |
| 'es_otherservices' | 280 | 228 | 81.429% |
| 'es_home' | 430 | 302 | 70.233% |
| 'es_health' | 2701 | 1696 | 62.792% |
| 'es_tech' | 315 | 158 | 50.159% |
| 'es_hyper' | 680 | 280 | 41.176% |
| 'es_wellnessandbeauty' | 1772 | 718 | 40.519% |
| 'es_barsandrestaurants' | 583 | 120 | 20.583% |
| 'es_fashion' | 592 | 116 | 19.595% |
| 'es_transportation' | 36768 | 0 | 0.000% |
| 'es_food' | 1961 | 0 | 0.000% |
| 'es_contents' | 62 | 0 | 0.000% |

Compare both transaction counts and fraud rates; rates are sample-based and should be interpreted with their denominators.

Figure: `../figures/category_fraud_patterns.png`.

## Fraud rates by step

Step-level transaction counts and fraud rates cover all 180 observed step values. Small per-step denominators can make rates variable; the chart shows counts alongside rates. Highest observed step rates are listed with their transaction counts:

| step | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 0.0 | 214.0 | 40.0 | 18.692% |
| 1.0 | 214.0 | 40.0 | 18.692% |
| 2.0 | 217.0 | 40.0 | 18.433% |
| 3.0 | 219.0 | 40.0 | 18.265% |
| 5.0 | 221.0 | 40.0 | 18.100% |
| 4.0 | 222.0 | 40.0 | 18.018% |
| 7.0 | 222.0 | 40.0 | 18.018% |
| 6.0 | 225.0 | 40.0 | 17.778% |
| 8.0 | 225.0 | 40.0 | 17.778% |
| 10.0 | 228.0 | 40.0 | 17.544% |

Figure: `../figures/fraud_rate_by_step.png`.

## Fraud rates by recorded age value

Age is analyzed using the literal categorical values present in the dataset, without assigning unsupported demographic meanings to the codes.

| age_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| '0' | 211 | 48 | 22.749% |
| '1' | 4829 | 689 | 14.268% |
| '2' | 15708 | 2344 | 14.922% |
| '3' | 12482 | 1755 | 14.060% |
| '4' | 9347 | 1410 | 15.085% |
| '5' | 5159 | 686 | 13.297% |
| '6' | 2174 | 261 | 12.006% |
| 'U' | 90 | 7 | 7.778% |

Figure: `../figures/fraud_rate_by_age_value.png`.

## Customer and merchant patterns

Transaction volume is descriptive context, not evidence of fraud. Tables show both counts and fraud rates; rate rankings require at least 20 transactions per entity to avoid highlighting rates based on very small denominators. All results remain specific to the fraud-enriched sample.

### Customer

Highest transaction volume (volume is context, not evidence of fraud):

| entity_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 'C1350963410' | 148 | 144 | 97.297% |
| 'C806399525' | 136 | 125 | 91.912% |
| 'C1849046345' | 131 | 127 | 96.947% |
| 'C1978250683' | 124 | 110 | 88.710% |
| 'C1275518867' | 123 | 114 | 92.683% |
| 'C2004941826' | 119 | 119 | 100.000% |
| 'C381007057' | 92 | 89 | 96.739% |
| 'C1572610482' | 89 | 89 | 100.000% |
| 'C1817318774' | 88 | 87 | 98.864% |
| 'C910454738' | 87 | 87 | 100.000% |

Highest fraud rates with at least 20 transactions:

| entity_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 'C2004941826' | 119 | 119 | 100.000% |
| 'C1572610482' | 89 | 89 | 100.000% |
| 'C910454738' | 87 | 87 | 100.000% |
| 'C483912131' | 80 | 80 | 100.000% |
| 'C617723960' | 40 | 40 | 100.000% |
| 'C928090253' | 21 | 21 | 100.000% |
| 'C1163388474' | 20 | 20 | 100.000% |
| 'C1817318774' | 88 | 87 | 98.864% |
| 'C1175850429' | 84 | 82 | 97.619% |
| 'C1350963410' | 148 | 144 | 97.297% |

### Merchant

Highest transaction volume (volume is context, not evidence of fraud):

| entity_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 'M1823072687' | 21702 | 0 | 0.000% |
| 'M348934600' | 15066 | 0 | 0.000% |
| 'M85975013' | 1961 | 0 | 0.000% |
| 'M480139044' | 1748 | 1634 | 93.478% |
| 'M980657600' | 1501 | 1472 | 98.068% |
| 'M855959430' | 680 | 280 | 41.176% |
| 'M151143676' | 583 | 120 | 20.583% |
| 'M732195782' | 526 | 518 | 98.479% |
| 'M1053599405' | 491 | 0 | 0.000% |
| 'M1198415165' | 451 | 358 | 79.379% |

Highest fraud rates with at least 20 transactions:

| entity_label | transactions | fraud_count | fraud_rate_percent |
| --- | --- | --- | --- |
| 'M1353266412' | 64 | 64 | 100.000% |
| 'M3697346' | 291 | 290 | 99.656% |
| 'M1873032707' | 217 | 216 | 99.539% |
| 'M1294758098' | 185 | 184 | 99.459% |
| 'M732195782' | 526 | 518 | 98.479% |
| 'M2011752106' | 169 | 166 | 98.225% |
| 'M980657600' | 1501 | 1472 | 98.068% |
| 'M857378720' | 94 | 92 | 97.872% |
| 'M2080407379' | 37 | 36 | 97.297% |
| 'M2122776122' | 209 | 200 | 95.694% |

Figure: `../figures/customer_merchant_patterns.png`. Each point represents an entity with the stated minimum support; the dashed line marks the sample-wide fraud rate.

## Figures

- `../figures/amount_by_fraud.png`
- `../figures/category_fraud_patterns.png`
- `../figures/fraud_rate_by_step.png`
- `../figures/fraud_rate_by_age_value.png`
- `../figures/customer_merchant_patterns.png`
