#!/usr/bin/env python3
"""
Extract Top Characters from SafeBooru Dataset
Filters all_data.csv to get top N characters with most images
Creates balanced dataset for high-accuracy multi-class training
"""

import pandas as pd
from pathlib import Path
import re
from collections import Counter

print("=" * 70)
print("🎯 EXTRACTING TOP CHARACTERS FROM SAFEBOORU DATA")
print("=" * 70)

# Character name variations to standardize
CHARACTER_MAPPINGS = {
    'uzumaki_naruto': 'naruto',
    'naruto_uzumaki': 'naruto',
    'monkey_d._luffy': 'luffy',
    'luffy': 'luffy',
    'gojo_satoru': 'gojo',
    'satoru_gojo': 'gojo',
    'son_goku_(dragon_ball)': 'goku',
    'goku': 'goku',
    'ryoumen_sukuna': 'sukuna',
    'sukuna': 'sukuna',
    
    # Additional popular characters
    'kurosaki_ichigo': 'ichigo',
    'ichigo_kurosaki': 'ichigo',
    'midoriya_izuku': 'deku',
    'izuku_midoriya': 'deku',
    'yeager_eren': 'eren',
    'eren_yeager': 'eren',
    'kamado_tanjirou': 'tanjiro',
    'tanjiro_kamado': 'tanjiro',
    'uchiha_sasuke': 'sasuke',
    'sasuke_uchiha': 'sasuke',
    'roronoa_zoro': 'zoro',
    'zoro': 'zoro',
    'vegeta': 'vegeta',
    'bakugou_katsuki': 'bakugo',
    'katsuki_bakugo': 'bakugo',
    'kamado_nezuko': 'nezuko',
    'nezuko_kamado': 'nezuko',
    'ackerman_mikasa': 'mikasa',
    'mikasa_ackerman': 'mikasa',
}

def extract_characters_from_tags(tags_str):
    """Extract character names from tag string"""
    if pd.isna(tags_str):
        return []
    
    tags = tags_str.lower().split()
    characters = []
    
    for tag in tags:
        # Check if tag matches any known character
        if tag in CHARACTER_MAPPINGS:
            characters.append(CHARACTER_MAPPINGS[tag])
        # Also check if tag contains character name
        for char_tag, char_name in CHARACTER_MAPPINGS.items():
            if char_tag in tag:
                characters.append(char_name)
                break
    
    return list(set(characters))  # Remove duplicates

def filter_dataset(csv_path='all_data.csv', top_n=15, min_images=500):
    """Filter dataset to top N characters"""
    
    print(f"\n📖 Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"   Total images: {len(df):,}")
    
    print("\n🔍 Extracting character tags...")
    df['characters'] = df['tags'].apply(extract_characters_from_tags)
    
    # Count character occurrences
    all_chars = []
    for char_list in df['characters']:
        all_chars.extend(char_list)
    
    char_counts = Counter(all_chars)
    
    print(f"\n📊 Top {top_n} characters found:")
    top_characters = []
    for i, (char, count) in enumerate(char_counts.most_common(top_n), 1):
        if count >= min_images:
            top_characters.append(char)
            print(f"   {i:2d}. {char:15s}: {count:6,} images")
    
    if len(top_characters) < top_n:
        print(f"\n⚠️  Only {len(top_characters)} characters have {min_images}+ images")
    
    # Filter to images containing at least one top character
    print(f"\n🔧 Filtering dataset to top {len(top_characters)} characters...")
    df['has_top_char'] = df['characters'].apply(
        lambda chars: any(c in top_characters for c in chars)
    )
    filtered = df[df['has_top_char']].copy()
    
    # Assign primary character (first match in top list)
    def get_primary_char(chars):
        for char in top_characters:
            if char in chars:
                return char
        return None
    
    filtered['primary_character'] = filtered['characters'].apply(get_primary_char)
    
    # Remove any rows without primary character
    filtered = filtered[filtered['primary_character'].notna()]
    
    print(f"   Filtered images: {len(filtered):,}")
    
    # Show distribution
    print(f"\n📈 Class distribution:")
    char_dist = filtered['primary_character'].value_counts()
    for char, count in char_dist.items():
        pct = 100 * count / len(filtered)
        bar = '█' * int(pct / 2)
        print(f"   {char:15s}: {count:6,} ({pct:5.1f}%) {bar}")
    
    # Save filtered dataset
    output_file = f'filtered_top{len(top_characters)}_chars.csv'
    filtered[['id', 'sample_url', 'tags', 'primary_character']].to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")
    
    return filtered, top_characters

if __name__ == '__main__':
    # Extract top 15 characters with at least 500 images each
    filtered_df, top_chars = filter_dataset(top_n=15, min_images=500)
    
    print("\n" + "=" * 70)
    print("✅ CHARACTER EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nDataset ready: filtered_top{len(top_chars)}_chars.csv")
    print(f"Total images: {len(filtered_df):,}")
    print(f"Classes: {len(top_chars)}")
    print(f"\nNext steps:")
    print(f"1. Download images using: python src/download_from_filtered.py")
    print(f"2. Update dataset.yaml with {len(top_chars)} classes")
    print(f"3. Train: python src/train_improved.py")
    print("=" * 70)
