# humor detection in Casual Conversations

The filtering process contains three steps:
1. Preprocessing
2. Summarization
3. Statement detection

Description:
- `script`: Code for building the knowledge graph
    1. Crawl from Reddit
    2. Filter the submissions and comments of Reddit
    3. Filter ED, and then merge with Reddit
    4. Cluster
    -askreddit_percen.py: one running script for whole processing, including percentage filter
    -gpt2.py:script for calculate suprisal and uncertainties
- `data`: All the data for building the knowledge graph
    - `ed`: First and second turns of ED after filtering
    - `reddit`: Reddit submissions and comments
        - `raw`: Raw data crawled using PushShift APIs
        - `filtered`: Submissions and comments after filtering
        - `matched`: Submissions and comments matched into single `csv` files
    - `merged`: Reddit and ED merged
- `plot`: Some visualizations

Due to the file size limitation of GitHub, some files are stored on [Google Drive](https://drive.google.com/drive/folders/17MaH7-uzIfIRac_PGLPnmwj7EVmARUKo?usp=sharing):

- Colbert
- `data/*`

