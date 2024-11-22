# README
 
MR-Eye atlas is a novel digital atlas constructed from MR images (T1-weighted MRI acquired at 1.5T ) of large-scale population of healthy volunteers.
It gathers a male, a female, and a combined (male and female) structural atlases constructed from 594 males and 616 females, with their corresponding probability maps of the different labels projected onto the average respective male and female templates.
Additionally, the maximum probability maps are provided for each case.

![alt text](./A_preview_figure.png)

## Authors

Jaime Barranco, Adrian Luyken, Philipp Stachs, Oscar Esteban, Yasser Aleman, Oliver Stachs, Sönke Langner, Benedetta Franceschiello and Meritxell Bach Cuadra.

## License
This work is distributed with License Creative Commons Attribution 4.0 International (see D_License.txt)

## Structure

We provide a .zip file containing:

- sub_metadata.csv: dataset summary table (SubjectID, Sex, Age, Height, Weight, BMI)
- template.nii.gz: atlas of the eye images (per sex and combined)
- max_prob_map.npy and max_prob_map.nii.gz: maximum probability maps (per sex and combined)
- prob_map.npy and prob_map.nii.gz: probability maps (per sex and combined)
- Colin27 [[1]](#ciric)[[2]](#fonov) T1w image + eye labels (also eye cropped image + labels)
- MNI152 [[1]](#ciric)[[3]](#holmes) T1w image + eye labels (also eye cropped image + labels)

T1w images retrieved from <https://github.com/templateflow/templateflow>.

```bash
.
└── eye_atlas
    ├── colin27
    │   ├── 0_tpl-MNIColin27_T1w_labels.nii.gz
    │   ├── 0_tpl-MNIColin27_T1w.nii.gz
    │   ├── 1_tpl-MNIColin27_T1w_cropped_labels.nii.gz
    │   └── 1_tpl-MNIColin27_T1w_cropped.nii.gz
    ├── combined
    │   ├── 0_template.nii.gz
    │   ├── 1_max_prob_map.nii.gz
    │   ├── 1_max_prob_map.npy
    │   ├── 2_prob_map.nii.gz
    │   └── 2_prob_map.npy
    ├── female
    │   ├── 0_template.nii.gz
    │   ├── 1_max_prob_map.nii.gz
    │   ├── 1_max_prob_map.npy
    │   ├── 2_prob_map.nii.gz
    │   └── 2_prob_map.npy
    ├── male
    │   ├── 0_template.nii.gz
    │   ├── 1_max_prob_map.nii.gz
    │   ├── 1_max_prob_map.npy
    │   ├── 2_prob_map.nii.gz
    │   └── 2_prob_map.npy
    ├── mni152
    │   ├── 0_tpl-MNI152NLin2009cAsym_res-01_T1w_labels.nii.gz
    │   ├── 0_tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz
    │   ├── 1_tpl-MNI152NLin2009cAsym_res-01_T1w_cropped_labels.nii.gz
    │   └── 1_tpl-MNI152NLin2009cAsym_res-01_T1w_cropped.nii.gz
    └── sub_metadata.csv
    
    6 directories, 24 files
```

## Dataset

The cohort was originally acquired within the Study of Health in Pomerania (SHIP) [^1] and reused in the context of this study.
A total of 3030 healthy subjects underwent whole-body MRI on a 1.5T scanner Magnetom Avanto (Siemens Medical Solutions, Erlangen, Germany) without contrast agent, from which we used 1210 subjects for this study.
There were 594 males and 616 females. Subjects were overall aged between 28 and 89 (56±13) years old.
T1-weighted (T1w) images of the head were acquired using a 12-channel head coil, 176 slices per volume, with a slice thickness of 1mm, and a field of view of 256mm, voxel size 1 mm3, TR=1900 ms, TI=1100 ms, TE=3.37 ms.
During the MRI examination, subjects rested their eyes naturally without specific guidelines for viewing or eyelid position. All participants gave informed written consent.
The study was approved by the Medical Ethics Committee of the University of Greifswald and followed the Declaration of Helsinki. All data of the study participants were accessed from an anonymized database.

## Template Construction

We performed metric-based registration, consisting of rigid, affine, and then deformable registration, with ANTs toolkit [^2] to iteratively create an average mapping of the subjects grouped by sex (594 males and 616 females).
We made use of the multivariate template construction tool, using as input images the right-eye-cropped ones obtained from the atlas-based segmentation method (in Supplementary Materials in the paper/preprint).
Therefore, they were much smaller than the initial ones (that included the whole head).
The maximum size of these right-eye-cropped images for the three axes were 61 x 70 x 68 and 77 x 95 x 94 voxels for the male and female case, respectively, and the size of the original images was 176 x 256 x 176 voxels.
The size of the voxels remained 1mm3.
For the deformable registration, we chose the SyN registration algorithm with the similarity metric of cross-correlation.
We chose four resolution levels (8, 4, 2, 1), and iterated over each level for 80, 60, 40, and 10 iterations, respectively.
Considering the reduced size of the images, we set the iteration limit (the number of iterations of the template construction) to 15, as we wanted to allow enough iterations for the template to converge and capture the variations present in our dataset.
We used a 11th Gen Intel® Core™ i9-11900K × 16 processor with 64GB of RAM.
The time spent to construct both atlases were 16h 15m 45s and 32h 16m 45s for the male and female cases, respectively.

```bash
#!/bin/bash

output_dir="/path/to/output_dir"
input_dir="/path/to/input_dir"

antsMultivariateTemplateConstruction2.sh \
    -d 3 \
    -o "$output_dir" \
    -i 15 \
    -g 0.2 \
    -j 16 \
    -c 2 \
    -k 1 \
    -w 1 \
    -n 1 \
    -r 1 \
    -m CC[2] \
    -q 80x60x40x10 \
    -f 8x4x2x1 \
    -s 3x2x1x0 \
    -t SyN \
    "$input_dir"*.nii.gz
```

## Probability Maps

To generate the labels on both eye atlases, we first registered them with each subject of its respective group (male or female), and project the labels obtained by the segmentation method, nnU-Net, of each subject to the atlas’ space.
The whole process lasted for 25m and 39m for males and females, respectively. We then created the maximum probability map of the labels for both atlases based on majority voting.
We also generated the probability maps of the labels for both atlases by adjusting the intensity of the color of each voxel per label based on its probability to belong to each one of the classes.
More precisely, we assigned an RGB color to every label, converted them to HSV, multiplied the S (saturation) and V (value) components of the color space by the probability per label, reconverted to RGB for visualization, and blended the resulting RGB values for the different labels.
This way, low-probability voxels (per label) will appear greyish, showing the uncertainty of those voxels belonging to a single class.

```python
""" Create maximum probability maps """

from collections import Counter

# Matrices for most likely tissue and probability of that tissue (divided by number of images)
stat_matrix = np.empty(image_shape)  # statistic matrix
prob_matrix = np.empty(image_shape)

# Probability calculation for each voxel
for x in range(image_shape[0]):
    for y in range(image_shape[1]):
        for z in range(image_shape[2]):
            voxel_values = voxel_arrays[x, y, z, :]
            freq = Counter(voxel_values)
            voxel_frequencies = freq.most_common()  # [(label, frequency), ...]
            # [0][0] most frequent value, [1][0] second most frequent value
            # [0][1] most frequent frequency, [1][1] second most frequent frequency
            voxel_value = voxel_frequencies[0][0]
            voxel_value_frequency = voxel_frequencies[0][1]
            stat_matrix[x, y, z] = voxel_value

# Save the entire max prob map to a .npy file
np.save(f'{maps_dir}/max_prob_map.npy', stat_matrix)

# Save max prob map as nifti
stat_nifti = nb.Nifti1Image(stat_matrix, sample_image.affine, sample_image.header)
nb.save(stat_nifti, f'{maps_dir}/max_prob_map.nii.gz')
```

```python
""" Compute probability of each structure per voxel """

num_labels = 10
probs = np.zeros((image_shape[0], image_shape[1], image_shape[2], num_labels))
for label in range(num_labels):
    probs[:, :, :, label] = np.count_nonzero(voxel_arrays == label, axis=3) / num_images

# Save the prob matrix to a .npy file
np.save(f'{maps_dir}/probs.npy', probs)
```

```python
""" Create ponderated RGB image for all subjects per label"""

import numpy as np
import nibabel as nb
import glob, os
import matplotlib.colors as mcolors
# matplotlib (https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py)

TYPE = 'female' # male, female, combined
METHOD = 'nnunet' # 'atlas' or 'nnunet'

labels_dir = f'/mnt/sda1/Repos/a-eye/Output/eye_model/{TYPE}/output/registrationToTemplate'
maps_dir = f'/mnt/sda1/Repos/a-eye/Output/eye_model/{TYPE}/output/maps/{METHOD}'

# Sample image to get the shape
sample_image_path = sorted(glob.glob(labels_dir + f'/*/labels_{METHOD}.nii.gz'))[0]
sample_image = nb.load(sample_image_path)
image_shape = sample_image.shape
print(image_shape)

# Define colors for the different labels (tissues)
colors = {
    0: [0, 0, 0],  # background - black
    1: [255, 0, 0],  # lens - tab:red
    2: [0, 255, 0],  # globe - green
    3: [0, 0, 255],  # optic nerve - tab:blue
    4: [255, 255, 0],  # intraconal fat - yellow
    5: [0, 255, 255],  # extraconal fat - cyan
    6: [255, 0, 255],  # lateral rectus muscle - magenta
    7: [144, 92, 44],  # medial rectus muscle - brown
    8: [255, 140, 0],  # inferior rectus muscle - orange
    9: [128, 0, 128],  # superior rectus muscle - purple
}

# Define names for the different labels (tissues)
label_names = {
    0: '0_background',
    1: '1_lens',
    2: '2_globe',
    3: '3_optic_nerve',
    4: '4_intraconal_fat',
    5: '5_extraconal_fat',
    6: '6_lateral_rectus_muscle',
    7: '7_medial_rectus_muscle',
    8: '8_inferior_rectus_muscle',
    9: '9_superior_rectus_muscle',
}

# Load probability matrix
# matrix = np.load('/mnt/sda1/Repos/a-eye/Output/eye_model/combined/output/maps/nnunet/voxel_arrays.npy')
# matrix_shape = matrix.shape
num_subjects = 1210  # matrix_shape[3]
probs = np.load(f'{maps_dir}/probs.npy')

# Colors to rgb using matplotlib
rgb_colors = {}
for key, value in colors.items():
    value = [x / 255 for x in value]  # normalize to [0, 1]
    rgb_colors[key] = mcolors.to_rgb(value) # no alpha channel
print("rgb: ", rgb_colors)

# Colors to hsv
hsv_colors = {}
for key, value in rgb_colors.items():
    hsv_colors[key] = mcolors.rgb_to_hsv(value)
print("hsv: ", hsv_colors)

# Create the output image
output_image_rgb = np.zeros((image_shape[0], image_shape[1], image_shape[2], 3)) # 3 channels

# Reduce intensity based on probabilities
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        for k in range(image_shape[2]):
            blended_rgb = np.zeros(3)
            for label in range(len(colors)):
                hsv = hsv_colors[label].copy()
                hsv[1] *= probs[i, j, k, label]  # Reduce intensity (Saturation)
                hsv[2] *= probs[i, j, k, label]  # Reduce intensity (Value)
                blended_rgb += mcolors.hsv_to_rgb(hsv)
            output_image_rgb[i, j, k] = blended_rgb

# Normalize to keep RGB values within [0, 1]
output_image_rgb = np.clip(output_image_rgb, 0, 1)

# Convert back to 0-255 range for visualization
output_image_rgb = (output_image_rgb * 255).astype(np.uint8)

# Save the output_rgb_image as a .npy file
np.save(f'{maps_dir}/prob_map.npy', output_image_rgb)

# Save the output_rgb_image as a nifti file
output_image_nifti = nb.Nifti1Image(output_image_rgb, sample_image.affine, sample_image.header)
nb.save(output_image_nifti, f'{maps_dir}/prob_map.nii.gz')
```

## Registration to common volumetric coordinate systems (VCS)

We first cropped the eye region of the templates [[2]](#fonov)[[3]](#holmes) using their right-eye masks that we extracted by a modified version of the `antsBrainExtraction.sh`.
Then, we registered them to the combined eye atlas, project its labels onto the cropped spaces, and finally transpose them back into the original spaces (inverse cropping).

## References

<a id="ciric"></a> 1. Ciric, R., Thompson, W. H., Lorenz, R., Goncalves, M., MacNicol, E., Markiewicz, C. J., Halchenko, Y. O., Ghosh, S. S., Gorgolewski, K. J., Poldrack, R. A., & Esteban, O. (2022). TemplateFlow: FAIR-sharing of multi-scale, multi-species brain models. bioRxiv. <https://doi.org/10.1101/2021.02.10.430678>

<a id="fonov"></a> 2. Fonov, V., Evans, A., McKinstry, R., Almli, C., & Collins, D. (2009). Unbiased nonlinear average age-appropriate brain templates from birth to adulthood. NeuroImage, 47, S102. <https://doi.org/10.1016/S1053-8119(09)70884-5>

<a id="holmes"></a> 3. Holmes CJ, Hoge R, Collins L, Woods R, Toga AW, Evans AC. “Enhancement of MR images using registration for signal averaging.” J Comput Assist Tomogr. 1998 Mar-Apr;22(2):324–33. <http://dx.doi.org/10.1097/00004728-199803000-00032>

<a id="schmidt"></a> 4. Schmidt, P., Kempin, R., Langner, S., Beule, A., Kindler, S., Koppe, T., Völzke, H., Ittermann, T., Jürgens, C., & Tost, F. (2019). Association of anthropometric markers with globe position: A population-based MRI study. PLoS ONE, 14(2), e0211817. <https://doi.org/10.1371/journal.pone.0211817>

<a id="avants"></a> 5. Avants, B., Tustison, N. J., & Song, G. (2009). Advanced Normalization Tools: V1.0. The Insight Journal. <https://doi.org/10.54294/uvnhin>.