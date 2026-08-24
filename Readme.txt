Purpose: pixel-level tumor segmentation on brain MRI (LGG dataset) — framed as dense classification, not just "AI project"
Architecture: the actual U-Net you built — encoder/decoder blocks, skip connections — described functionally
Data pipeline: the synced image/mask augmentation generator
Training setup: Dice loss, IoU/Dice metrics, Adam, the three callbacks
Results: real numbers pulled from your training log — ~0.79 val IoU, ~0.88 val Dice, ~99.7% pixel accuracy at 100 epochs — much stronger than a vague "achieved good results" line
Tech stack: TensorFlow, Keras, Deep Learning, Computer Vision, CNNs, Medical Image Segmentation