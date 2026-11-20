# Data

## `20newsgroups/20news-bydate.tar.gz`

The **20 Newsgroups** corpus, "bydate" version: 18,846 Usenet posts in 20 newsgroups, split by date
into train (60%) and test (40%), with duplicates and some headers already removed.

| | |
|---|---|
| Source | Jason Rennie's 20 Newsgroups page; the same file as scikit-learn's mirror (https://ndownloader.figshare.com/files/5975967) |
| Size | 14 MB compressed |
| SHA-256 | `8f1b2514ca22a5ade8fbb9cfa5727df95fa587f4c87b786e15c759fa66d95610` (checked on load) |

Load it with:

```bash
cd CODE && python scripts/01_ingest.py --archive ../DATA/20newsgroups/20news-bydate.tar.gz
```

**Preprocessing** (in `CODE/pipeline/ingest/`):
- Headers, quoted replies and signatures are stripped (the same rules as scikit-learn's `remove=`).
- Posts under 20 words are dropped, as are uuencoded or base64 attachments.
- Train and test are clustered together, and the newsgroup labels are used only for scoring.
- 17,090 posts remain. The cloud database used by the dashboards holds a 10,305-post subset.
