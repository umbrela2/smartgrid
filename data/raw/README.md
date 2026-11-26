# Raw input files (not included in this repository)

Download the Kaggle dataset **"Smart meters in London"** (`jeanmidev/smart-meters-in-london`,
https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london) and copy these three files here:

| File | Used for |
|------|----------|
| `daily_dataset.csv` | one row per household (`LCLid`) per `day`: `energy_sum`, `energy_count` |
| `weather_daily_darksky.csv` | daily `humidity`, `windSpeed` (column `time` is the date) |
| `uk_bank_holidays.csv` | two columns: date, holiday type |

They are git-ignored on purpose (large, third-party data). Check the dataset's licence on Kaggle
before redistributing it.
