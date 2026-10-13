import pandas as pd, json
df = pd.read_parquet('/home/pinion/jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable/train-00000-of-00001.parquet')
print('cols:', list(df.columns))
print('rows:', len(df))
row = df.iloc[0].to_dict()
s = json.dumps(row, ensure_ascii=False, default=str)
print(s[:1500])
if 'kind' in df.columns:
    print('kind:', df['kind'].value_counts().to_dict())
if 'source' in df.columns:
    print('source:', df['source'].value_counts().head(8).to_dict())
if 'target' in df.columns:
    print('target sample:', df['target'].head(5).tolist())
    # soft targets?
    soft = sum(1 for t in df['target'] if isinstance(t, (list, tuple)))
    print('soft-target rows:', soft, '/', len(df))
