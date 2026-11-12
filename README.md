# Chlorella tractability

Code for the pub *Chlorella vulgaris phenotypes are structured and predictable across environmental conditions* [TO FILL: pub link].

We used high-throughput, label-free imaging to measure dozens of morphological features in 91 UV-mutagenized *Chlorella vulgaris* strains across six growth conditions (three temperatures × two growth stages). We then asked whether those phenotypes are reproducible, structured, and predictable in held-out environments.

## Repository layout

| Folder | What it contains |
|---|---|
| [`main_analyses/`](main_analyses/) | The notebook that reproduces every quantitative result and data figure in the pub, the helper module it uses, and the conda environments. **Start here.** |
| [`segmentation/`](segmentation/) | Image segmentation and feature extraction: raw microscopy images → the `zarr` image stores and `jsonl` feature files used by the main analysis. |
| [`background/`](background/) | Zoogle ortholog analysis behind choosing *Chlorella vulgaris* as a candidate organism. |

## Reproducing the pub

Follow [`main_analyses/README.md`](main_analyses/README.md). It covers setting up the environment, downloading the data from the BioImage Archive ([TO FILL: accession]), where to put it, and running the notebook.

The segmented data are already on the BioImage Archive, so you don't need to rerun segmentation to reproduce the results.

## Data

Segmented images (`zarr`) and per-cell features (`jsonl`) for all 7 experiments: BioImage Archive, [(https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD4378)].
