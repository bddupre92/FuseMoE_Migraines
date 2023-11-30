# Notes on MIMIC-IV data/processing/etc.

### Data
Tasks we discussed:

| IHM using the first 48 hours of data for patients who spent at least 48 hours in the ICU |  |
| --- | --- |
| Selection criteria | length-of-stay (in the ICU) >- 48 hours |
| Data included | first 48 hours of vitals/labs |
| Data files | `*_ihm-48_stays.pkl` where `*` is either `train`, `val`, or `test` |
| Running the task | set `--task` to `ihm-48` in `run_mimiciv.sh` |

| IHM with no restrictions |  |
| --- | --- |
| Selection criteria | Any ICU stay of any length |
| Data included | All labs/vitals for the ICU stay |
| Data files | `*_ihm-all_stays.pkl` |
| Running the task | set `--task` to `ihm-all` in `run_mimiciv.sh` |

Each of the pickle files contains time series (labs + vitals that were included in the NPJ paper) and radiological notes for the specified time frame (within 48 hours for `ihm-48` or at any point during ICU stay for `all-stays`). Additionally, only patients for whom there is *at least* one lab/vital and *at least* one radiological note within the specified time point are included.

Additionally, `--file_path` in `run_mimiciv.sh` should link the folder that contains the `.pkl` files.

**Additional notes** I'm also working on a version of the data files that contain `cxr` densenet embeddings. Will add to the Google Drive once I have some sort of working version of the pipeline that uses these embeddings (analgous to how text embeddings are handled.)