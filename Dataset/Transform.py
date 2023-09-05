import albumentations as A

train_transform = A.Compose([
    A.Resize(width = 224, height = 224),
    # A.HorizontalFlip(p = 0.5),
    # A.Rotate(limit = (-10,10), p = 0.6),
    A.RandomBrightnessContrast(contrast_limit = 0.05, brightness_limit = 0.05, p = 0.75),
])

test_transform = A.Compose([
    A.Resize(width = 224, height = 224),  # Thay đổi kích thước ảnh
])