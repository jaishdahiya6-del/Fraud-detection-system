# Dataset

Download the **Credit Card Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/mlg-ulb/creditcardfraud

Place the file here as:

```
data/creditcard.csv
```

Expected columns: `Time`, `V1`...`V28` (PCA features), `Amount`, `Class`
(`Class`: 1 = Fraud, 0 = Genuine).

No file is bundled in this repo due to size/licensing — the pipeline reads
whatever CSV you place here, so all results in this project are computed
from real data, not hardcoded.
