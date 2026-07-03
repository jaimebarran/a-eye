# Release Notes

## 2026-07-03 - Journal Publication & Pretrained Model Weights

### Added

- **Published Journal Paper**:  
  The associated manuscript is now published in *PLOS ONE*: *A-eye: Automated 3D MRI Segmentation and Morphometric Feature Extraction for Eye and Orbit Atlas Construction*. <https://doi.org/10.1371/journal.pone.0352257>

- **Pretrained Model Weights**:  
  Added `A-eye_nnUNet_model_weights.zip`, the pretrained nnU-Net weights (`Task313_Eye`) used to generate the segmentations underlying this atlas.

### Documentation Updates

- Updated `B_README.md` and `B_README.txt`:
  - Replaced the bioRxiv preprint link with the published *PLOS ONE* citation.
  - Added a new "Code, Pretrained Model & Web Platform" section linking to the [A-eye GitHub repository](https://github.com/jaimebarran/a-eye) and the [web platform](https://aeye.hevs.ch).
  - Added instructions for installing the pretrained weights or running the ready-to-use [Docker image](https://hub.docker.com/r/jaimebarran/fw_gear_aeye) (`jaimebarran/fw_gear_aeye`).

## 2026-04-17 - Manual Segmentation Update

### Added

- **Combined Atlas Manual Segmentation**:  
  Added `C_eye_atlas/combined/3_manual_seg.nii.gz`, a manual segmentation for the combined eye atlas.

- **ITK-SNAP Segmentation File**:  
  Added `C_eye_atlas/combined/3_manual_seg.itksnap` to support inspection and editing of the manual segmentation in ITK-SNAP.

### Documentation Updates

- Updated `B_README.md` and `B_README.txt` to describe the new manual segmentation files.
- Updated the documented archive structure for `C_eye_atlas/combined` and changed the file count from 24 to 26 files.

Updated `C_eye_atlas/combined` structure:

```bash
combined
├── 0_template.nii.gz
├── 1_max_prob_map.nii.gz
├── 1_max_prob_map.npy
├── 2_prob_map.nii.gz
├── 2_prob_map.npy
├── 3_manual_seg.itksnap
└── 3_manual_seg.nii.gz
```

## 2025-07-15 - Version 2 Zenodo Release

### Added

- **Combined Eye Atlas**:  
  Added the combined male and female eye atlas, including the template, maximum probability map, and probability map.

- **Label Projections**:  
  Added labels projected onto:
  - Colin27 space (both images and cropped versions).
  - MNI152 T1-weighted (T1w) space (both images and cropped versions).

- **Additional Preview Figures**:  
  - Labels overlaid onto Colin27 and MNI152 images.
  - Schematic diagram illustrating the processing workflow.

### Documentation Updates

- Updated the README files to describe the combined eye atlas and label projections.
- Added release notes in Markdown and text formats.

## 2024-08-15 - Initial Zenodo Release

### Added

- **Female Eye Atlas**:  
  Released the female eye atlas, including its template, maximum probability map, and probability map.

- **Male Eye Atlas**:  
  Released the male eye atlas, including its template, maximum probability map, and probability map.

- **Dataset Metadata**:  
  Included `sub_metadata.csv` with subject-level summary information.
