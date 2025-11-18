# DullRazor Hair Removal for Dermoscopic Images
A Preprocessing Script for Skin Lesion Datasets (Python + OpenCV)

A simple way to remove hair from skin lesion images. When working with skin cancer datasets (like ISIC or HAM10000), many images contain hair.
Hair can cover the lesion and make it harder for doctors and machine learning models to see important details.
The script processes an entire dataset with class subfolders and outputs hair-removed images while preserving the original folder structure.

Why remove hair?
Hair on the skin can:
<li>hide the true shape of the lesion
<li>confuse machine learning models
<li>make the image look noisy or unclear
<li>affect accuracy during training

By removing hair, the images become cleaner, easier to analyze, and better for training AI models.

# How DullRazor actually works
DullRazor follows a few simple steps:

1. Turn the image to grayscale
→ easier to detect dark hair.

2. Find the dark hair using a filter
→ this highlights hair areas.

3. Create a mask of where the hair is
→ white = hair, black = not hair.

4. Make the mask a bit thicker (optional)
→ so all the hair is removed.

5. Use inpainting to fill the hair areas
→ fills the hair pixels using nearby skin pixels
→ makes the image look natural.
