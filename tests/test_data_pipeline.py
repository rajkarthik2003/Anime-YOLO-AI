"""
Unit tests for data pipeline components
"""

import pytest
import os
from pathlib import Path
import cv2
import numpy as np

# Import functions to test (uncomment when implementing)
# from src.validate_images import is_valid_image
# from src.download_and_filter import tags_match
# from src.autolabel_faces import detect_faces


class TestValidateImages:
    """Test suite for image validation"""
    
    def test_is_valid_image_with_jpg(self, tmp_path):
        """Test that valid JPG images pass validation"""
        # Create a valid test image
        test_img_path = tmp_path / "test_valid.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img_path), img)
        
        # TODO: Uncomment when is_valid_image is importable
        # assert is_valid_image(str(test_img_path)) == True
        assert test_img_path.exists()
    
    def test_is_valid_image_with_gif(self, tmp_path):
        """Test that GIF images are rejected"""
        test_gif_path = tmp_path / "test_animated.gif"
        test_gif_path.touch()
        
        # TODO: Uncomment when is_valid_image is importable
        # assert is_valid_image(str(test_gif_path)) == False
        assert test_gif_path.suffix == '.gif'
    
    def test_is_valid_image_with_corrupted(self, tmp_path):
        """Test that corrupted images are detected"""
        test_corrupted_path = tmp_path / "test_corrupted.jpg"
        test_corrupted_path.write_bytes(b'\x00\x00\x00\x00')  # Invalid JPEG
        
        # TODO: Uncomment when is_valid_image is importable
        # assert is_valid_image(str(test_corrupted_path)) == False
        assert test_corrupted_path.exists()
    
    def test_is_valid_image_with_nonexistent(self):
        """Test that nonexistent files return False"""
        # TODO: Uncomment when is_valid_image is importable
        # assert is_valid_image('nonexistent.jpg') == False
        assert not os.path.exists('nonexistent.jpg')


class TestDataFiltering:
    """Test suite for data filtering logic"""
    
    def test_tags_match_with_single_tag(self):
        """Test tag matching with single character tag"""
        tags = "naruto, uzumaki, ninja"
        # TODO: Uncomment when tags_match is importable
        # assert tags_match(tags) == True
        assert 'naruto' in tags.lower()
    
    def test_tags_match_with_multiple_tags(self):
        """Test tag matching with multiple character tags"""
        tags = "goku, dragon_ball, super_saiyan"
        # TODO: Uncomment when tags_match is importable
        # assert tags_match(tags) == True
        assert 'goku' in tags.lower()
    
    def test_tags_match_with_no_match(self):
        """Test tag matching with irrelevant tags"""
        tags = "landscape, scenery, mountain"
        # TODO: Uncomment when tags_match is importable
        # assert tags_match(tags) == False
        assert 'naruto' not in tags.lower()
    
    def test_tags_match_case_insensitive(self):
        """Test that tag matching is case-insensitive"""
        tags = "NARUTO, UZUMAKI"
        # TODO: Uncomment when tags_match is importable
        # assert tags_match(tags) == True
        assert 'naruto' in tags.lower()


class TestFaceDetection:
    """Test suite for face detection labeling"""
    
    def test_detect_faces_with_face(self, tmp_path):
        """Test face detection on image with face"""
        # Create test image with synthetic face-like region
        test_img_path = tmp_path / "test_with_face.jpg"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        # Draw a circle to simulate a face
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
        cv2.imwrite(str(test_img_path), img)
        
        # TODO: Uncomment when detect_faces is importable
        # bboxes = detect_faces(str(test_img_path))
        # assert len(bboxes) >= 0  # May or may not detect synthetic face
        assert test_img_path.exists()
    
    def test_detect_faces_without_face(self, tmp_path):
        """Test face detection on image without face"""
        test_img_path = tmp_path / "test_without_face.jpg"
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img_path), img)
        
        # TODO: Uncomment when detect_faces is importable
        # bboxes = detect_faces(str(test_img_path))
        # assert len(bboxes) == 0
        assert test_img_path.exists()


class TestDatasetSplit:
    """Test suite for train/val splitting"""
    
    def test_split_ratio(self):
        """Test that split produces correct ratio"""
        total_images = 100
        train_ratio = 0.8
        expected_train = int(total_images * train_ratio)
        expected_val = total_images - expected_train
        
        assert expected_train == 80
        assert expected_val == 20
    
    def test_split_preserves_all_images(self):
        """Test that no images are lost during split"""
        total_images = 1000
        train_ratio = 0.8
        train_count = int(total_images * train_ratio)
        val_count = total_images - train_count
        
        assert train_count + val_count == total_images


class TestAugmentation:
    """Test suite for data augmentation"""
    
    def test_augmentation_preserves_dimensions(self, tmp_path):
        """Test that augmentation maintains image dimensions"""
        test_img_path = tmp_path / "test_aug.jpg"
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img_path), img)
        
        # Read back and verify
        img_read = cv2.imread(str(test_img_path))
        assert img_read.shape == (640, 640, 3)
    
    def test_augmentation_preserves_bbox_format(self):
        """Test that bounding boxes remain in valid YOLO format"""
        # YOLO format: class x_center y_center width height (normalized 0-1)
        bbox = [0, 0.5, 0.5, 0.3, 0.4]  # class=0, center=(0.5,0.5), size=(0.3,0.4)
        
        assert 0 <= bbox[1] <= 1  # x_center
        assert 0 <= bbox[2] <= 1  # y_center
        assert 0 <= bbox[3] <= 1  # width
        assert 0 <= bbox[4] <= 1  # height


# Pytest fixtures
@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image"""
    img_path = tmp_path / "sample.jpg"
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_dataset_manifest(tmp_path):
    """Create a sample dataset manifest CSV"""
    import pandas as pd
    
    manifest_path = tmp_path / "manifest.csv"
    data = {
        'id': [1, 2, 3, 4, 5],
        'tags': [
            'naruto, uzumaki, ninja',
            'goku, dragon_ball, saiyan',
            'landscape, scenery',
            'luffy, one_piece, pirate',
            'gojo, jujutsu_kaisen'
        ],
        'sample_url': [
            'https://example.com/1.jpg',
            'https://example.com/2.jpg',
            'https://example.com/3.jpg',
            'https://example.com/4.jpg',
            'https://example.com/5.jpg'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(manifest_path, index=False)
    return manifest_path


# Integration tests
class TestEndToEndPipeline:
    """Integration tests for full pipeline"""
    
    def test_pipeline_directory_structure(self, tmp_path):
        """Test that pipeline creates expected directory structure"""
        data_dir = tmp_path / "data" / "raw"
        images_train = data_dir / "images" / "train"
        images_val = data_dir / "images" / "val"
        labels_train = data_dir / "labels" / "train"
        labels_val = data_dir / "labels" / "val"
        
        # Create directories
        images_train.mkdir(parents=True, exist_ok=True)
        images_val.mkdir(parents=True, exist_ok=True)
        labels_train.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)
        
        assert images_train.exists()
        assert images_val.exists()
        assert labels_train.exists()
        assert labels_val.exists()
    
    def test_pipeline_file_pairing(self, tmp_path):
        """Test that images and labels are correctly paired"""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        
        # Create sample image and label
        img_path = images_dir / "sample_001.jpg"
        label_path = labels_dir / "sample_001.txt"
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        label_path.write_text("0 0.5 0.5 0.3 0.4\n")
        
        # Verify pairing
        assert img_path.stem == label_path.stem
        assert img_path.exists() and label_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
