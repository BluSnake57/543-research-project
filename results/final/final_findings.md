# Sentinel-X Final Experimental Findings

## Objective

Sentinel-X investigates whether LLM-derived semantic information can improve early social-media virality prediction beyond propagation structure alone.

The system combines Qwen3.5-9B semantic analysis with GraphSAGE propagation modeling.

## Dataset Processing

Twitter15:
- 1,490 original cascades
- 1,487 valid cascades after graph validation/repair
- Median cascade size: 260 posts
- 75th percentile: 464 posts

Twitter16:
- 818 original cascades
- 816 valid cascades
- Median cascade size: 271 posts
- 75th percentile: approximately 474 posts

The final experiments use thresholds of 464 posts for Twitter15 and 475 posts for Twitter16.

Predictions use only propagation observed during the first 60 minutes.

## Graph-Only Baseline

Twitter15 GraphSAGE:
- F1: 76.79%
- Accuracy: 88.34%
- PR-AUC: 85.54%

Twitter16 GraphSAGE:
- F1: 73.68%
- Accuracy: 87.80%
- PR-AUC: 77.04%

## LLM Semantic Features

Qwen3.5-9B extracted five semantic signals:

1. Urgency
2. Sensationalism
3. Emotional language
4. Authority framing
5. Conspiracy framing

Features were extracted for all 2,308 source tweets across Twitter15 and Twitter16.

## Five-Feature Fusion

Combining all five Qwen features did not improve GraphSAGE.

Twitter15:
- Graph-only F1: 76.79%
- All-five fusion F1: 74.07%

Twitter16:
- Graph-only F1: 73.68%
- All-five fusion F1: 73.68%

This indicates that indiscriminately adding semantic features can introduce noise.

## Twitter15 Ablation Study

Urgency:
- F1: 75.23%

Sensationalism:
- F1: 75.21%

Emotional language:
- F1: 79.28%

Authority framing:
- F1: 75.00%

Conspiracy framing:
- F1: 78.18%

Emotional language produced the strongest Twitter15 result.

Compared with graph-only GraphSAGE:

- F1 increased from 76.79% to 79.28%.
- Accuracy increased from 88.34% to 89.69%.
- Precision increased from 76.79% to 80.00%.
- Recall increased from 76.79% to 78.57%.

Conspiracy framing also improved F1 to 78.18%.

## Cross-Dataset Validation

Emotional-language fusion was tested independently on Twitter16.

Graph-only:
- F1: 73.68%
- PR-AUC: 77.04%

Graph + emotional language:
- F1: 73.68%
- PR-AUC: 77.27%

The classification result was unchanged, although PR-AUC improved slightly.

Therefore, the Twitter15 improvement did not generalize as an F1 improvement to Twitter16.

## Main Finding

LLM-derived semantic information can complement propagation-based virality prediction, but its usefulness is feature- and dataset-dependent.

Emotional-language information improved GraphSAGE on Twitter15, while the same feature did not improve Twitter16 F1. Combining all five semantic features did not improve either dataset.

The results demonstrate that semantic feature selection is more important than simply adding more LLM-derived information.

## Live Sentinel-X Demonstration

The final Sentinel-X application implements:

Qwen3.5-9B
? semantic feature extraction
? GraphSAGE propagation analysis
? deterministic virality result

A Twitter15 demonstration cascade produced:

- Viral probability: 75.31%
- Decision cutoff: 71.28%
- Prediction: VIRAL
- Observation window: 60 minutes
- Target: at least 464 recorded posts
- Semantic feature: emotional_language

The live application therefore demonstrates genuine LLM-GNN integration rather than an LLM-generated virality estimate.

## Limitations

- Virality is defined using recorded cascade size rather than worldwide lifetime reach.
- Semantic improvements were not consistent across both datasets.
- Twitter15 and Twitter16 are historical datasets.
- Qwen semantic scores are model-derived features rather than human annotations.
- The experiments evaluate five selected semantic dimensions and do not establish that other textual representations would behave similarly.
- Graph preprocessing repairs malformed propagation structures, and these repairs may influence downstream graph features.

## Conclusion

Propagation structure remains a strong predictor of cascade virality. Selected LLM-derived semantic signals can add useful information, but semantic augmentation does not automatically improve performance.

Sentinel-X demonstrates a practical hybrid architecture in which an LLM analyzes content while a GNN models propagation structure, allowing both semantic and network information to contribute to early virality prediction.
