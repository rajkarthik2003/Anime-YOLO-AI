#!/usr/bin/env python3
"""
Expand Dataset to Detect More Anime Characters
Downloads and prepares additional character data from SafeBooru
"""

import pandas as pd
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

# Popular anime characters to add (expand the 5 we have)
ADDITIONAL_CHARACTERS = [
    # Shonen Jump characters
    'ichigo_kurosaki',  # Bleach
    'monkey_d._luffy',  # Already have, but more data
    'roronoa_zoro',     # One Piece
    'portgas_d._ace',   # One Piece
    'sasuke_uchiha',    # Naruto
    'kakashi_hatake',   # Naruto
    
    # My Hero Academia
    'midoriya_izuku',
    'bakugou_katsuki',
    'todoroki_shouto',
    'uraraka_ochako',
    
    # Attack on Titan
    'eren_yeager',
    'mikasa_ackerman',
    'levi_(shingeki_no_kyojin)',
    
    # Demon Slayer
    'kamado_tanjirou',
    'kamado_nezuko',
    'agatsuma_zenitsu',
    
    # Jujutsu Kaisen (already have Gojo)
    'itadori_yuuji',
    'fushiguro_megumi',
    'kugisaki_nobara',
    
    # Dragon Ball (already have Goku)
    'vegeta',
    'son_gohan',
    'trunks_(dragon_ball)',
    
    # Death Note
    'yagami_light',
    'l_(death_note)',
    
    # Fullmetal Alchemist
    'edward_elric',
    'alphonse_elric',
    
    # One Punch Man
    'saitama_(one-punch_man)',
    'genos_(one-punch_man)',
    
    # Hunter x Hunter
    'gon_freecss',
    'killua_zoldyck',
    
    # Chainsaw Man
    'denji_(chainsaw_man)',
    'power_(chainsaw_man)',
]

print("=" * 70)
print("🎯 ANIME CHARACTER DATASET EXPANSION")
print("=" * 70)
print(f"\nCurrent classes: 5 (Naruto, Luffy, Gojo, Goku, Sukuna)")
print(f"Potential classes: {len(set(ADDITIONAL_CHARACTERS)) + 5}")
print(f"New characters to add: {len(set(ADDITIONAL_CHARACTERS))}")

def download_character_data(character_tag, max_images=1000):
    """Download images for a specific character from SafeBooru"""
    print(f"\n📥 Downloading: {character_tag} (max {max_images} images)")
    
    # SafeBooru API endpoint
    base_url = "https://safebooru.org/index.php"
    params = {
        'page': 'dapi',
        's': 'post',
        'q': 'index',
        'tags': character_tag,
        'limit': 100,
        'pid': 0
    }
    
    # Create directory
    char_name = character_tag.replace('_', '-')
    img_dir = Path(f'data/expanded/{char_name}/images')
    img_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    # Note: This is simplified - actual implementation would need proper API handling
    print(f"   ⚠️  Manual download required: https://safebooru.org/index.php?page=post&s=list&tags={character_tag}")
    
    return downloaded

def prepare_for_yolo():
    """Prepare expanded dataset for YOLO training"""
    print("\n" + "=" * 70)
    print("📋 PREPARING DATASET FOR YOLO")
    print("=" * 70)
    
    # Read current dataset
    df = pd.read_csv('all_data.csv')
    print(f"\nCurrent dataset: {len(df)} images, 5 classes")
    
    # Show class distribution
    print("\nCurrent class distribution:")
    for char in df['character'].value_counts().head(10).items():
        print(f"  {char[0]:20s}: {char[1]:6d} images")

if __name__ == '__main__':
    prepare_for_yolo()
    
    print("\n" + "=" * 70)
    print("📝 NEXT STEPS TO EXPAND DATASET")
    print("=" * 70)
    print("""
1. **Choose Target Characters** (Start with 10-20 most popular)
   - Focus on iconic characters with distinct appearances
   - Avoid characters that look too similar

2. **Download More Data**
   - SafeBooru: https://safebooru.org/
   - Danbooru: https://danbooru.donmai.us/
   - Anime face datasets: AnimeFace, Tagged Anime Illustrations
   
3. **Annotate New Data**
   - Use existing Haar cascade to auto-detect faces
   - Manual review with LabelImg or CVAT
   - Or use our existing annotation pipeline: python src/data_prep.py

4. **Update dataset.yaml**
   - Add new class names
   - Update nc (number of classes)

5. **Retrain Model**
   - python src/train_improved.py
   - Larger model recommended for more classes (YOLOv8l or YOLOv8x)

6. **Expected Performance**
   - 10 classes: ~25-35% mAP (good)
   - 20 classes: ~20-30% mAP (acceptable)
   - 50+ classes: ~15-25% mAP (challenging but possible)
    """)
    
    print("\n💡 RECOMMENDATIONS:")
    print("   - Start with 10-15 iconic characters")
    print("   - Use YOLOv8 Large (43.7M params) for multi-class")
    print("   - Aim for 500+ images per character (balanced dataset)")
    print("   - Use stronger data augmentation")
    print("=" * 70)
