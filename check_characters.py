import pandas as pd

df = pd.read_csv('all_data.csv')
print('='*60)
print('TOP 30 CHARACTERS IN YOUR DATASET')
print('='*60)
counts = df['character'].value_counts().head(30)
for char, count in counts.items():
    print(f'{char:40s}: {count:6d} images')

print('\n' + '='*60)
print(f'Total unique characters: {df["character"].nunique()}')
print(f'Total images: {len(df)}')
