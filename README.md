# <img src="figs/A-eye_logo_square.png" width="40" style="vertical-align:middle;"> 👁️ A-eye: Automated 3D MRI Segmentation and Morphometric Feature Extraction for Eye and Orbit Atlas Construction

## Short title: *A-eye: Automated MRI Segmentation and Morphometry of the Eye*

---

![Overview of A-eye results](figs/Fig1.png)

*Figure 1. Overview of the A-eye results for automated MRI eye segmentation and morphometry.*

![Axial length automatic extraction](figs/Fig9.png)

*Figure 2. Example of axial length automatic extraction in 3D.*

## 📘 Overview

This repository accompanies the paper:

> **Barranco J.**, Luyken A., Jia Y., Kebiri H., Stachs P., Gordaliza P. M., Esteban O., Aleman Y., Sznitman R., Stachs O., Langner S., Franceschiello B.†, Bach Cuadra M.†  
> **A-eye: Automated 3D MRI Segmentation and Morphometric Feature Extraction for Eye and Orbit Atlas Construction**  
> † Equal last authorship
> 📧 *Corresponding authors:*  
> <jaime.barranco-hernandez@chuv.ch>, <benedetta.franceschiello@hevs.ch>, <meritxell.bachcuadra@unil.ch>  

---

## 🧩 Abstract

In this study, we introduce **an automated 3D segmentation of the healthy human adult eye and orbit from Magnetic Resonance Images (MRI)**, aimed at advancing ophthalmic diagnostics and treatments.  

Previous works have typically relied on small sample sizes and heterogeneous imaging modalities. Here, we leverage a **large-scale dataset of T1-weighted MRI scans from 1,245 subjects** and employ the **deep learning-based nnU-Net** for MR-Eye segmentation tasks.  

Our results demonstrate **robust and accurate 3D segmentations** of the lens, globe, optic nerve, rectus muscles, and orbital fat. We further provide **automated morphometric biomarkers** such as axial length and volumetric measures, as well as benchmarking analyses correlating body mass index with eye structure volumes.  

Quality control protocols ensure the **reliability of large-scale segmentation**, enhancing the applicability of our pipeline in clinical research.  
A major outcome of this work is the **first large-scale, unbiased eye atlases (female, male, and combined)** to promote standardization of spatial normalization tools for MR-Eye.

---

## 🧠 Keywords

`MRI` · `Eye` · `MR-Eye` · `3D segmentation` · `Large-scale dataset` · `Imaging biomarkers` · `Morphometry` · `Ophthalmology` · `Atlas` · `Benchmarking` · `Axial length`

---

## 🏛️ Affiliations

1. CIBM Center for Biomedical Imaging, Lausanne, Switzerland  
2. Department of Radiology, Lausanne University Hospital (CHUV) and University of Lausanne (UNIL), Lausanne, Switzerland  
3. HES-SO University of Applied Sciences and Arts Western Switzerland  
4. The Sense Innovation and Research Center, Lausanne and Sion, Switzerland  
5. Department of Ophthalmology, Rostock University Medical Center, Rostock, Germany  
6. Karlsruhe Institute of Technology (KIT)  
7. ARTORG Center for Biomedical Engineering, University of Bern, Bern, Switzerland  
8. Department Life, Light & Matter, University of Rostock, Rostock, Germany  
9. Institute for Diagnostic and Interventional Radiology, Pediatric and Neuroradiology, Rostock University Medical Center, Rostock, Germany  

† Equal last authorship

---

## 🌍 Web Platform

🚀 **Coming soon!**

An interactive web interface will allow exploration of the MR-Eye atlases, morphometric statistics, and automated segmentation examples.

---

## 📚 How to Cite

If you use the A-eye pipeline, atlas, or derived datasets in your work, please cite the relevant resources below:

### 🧭 Eye Atlases Dataset

> Barranco J, Luyken A, Stachs P, Esteban O, Aleman-Gomez Y, Stachs O, *et al.*  
> **MR-Eye atlas: a large-scale atlas of the eye based on T1-weighted MR imaging** [dataset].  
> *Zenodo*, 2024.  
> [https://doi.org/10.5281/zenodo.13325371](https://doi.org/10.5281/zenodo.13325371)

![MR-Eye atlases (female, male, combined) and in common VCS](figs/Fig6.png)

*Figure 3. Visualization of the MR-Eye population atlases (female, male, and combined) and in common VCS.*    

### 🧩 Proceedings Paper

> Barranco Hernandez J, Luyken A, Stachs O, Langner S, Franceschiello B, Bach Cuadra M.  
> **A-eye: automated 3D segmentation of healthy human eye and orbit structures and axial length extraction.**  
> 2025.  
> [https://doi.org/10.26039/TA7F-X088](https://doi.org/10.26039/TA7F-X088)

### 📰 Journal Paper

> *Coming soon!*

---

## 🧠 Repository Structure

- `A-eye/`
  - `models/` — Trained nnU-Net weights and configs (when released)
  - `notebooks/` — Example scripts for segmentation and morphometry
  - `data/` — Sample data or download scripts (if applicable)
  - `docs/` — Documentation and supplementary material
      - `LICENSE` — License file

---

## 🧑‍💻 Contact

For questions or collaborations, please contact:

- Jaime Barranco — [jaime.barranco-hernandez@chuv.ch](mailto:jaime.barranco-hernandez@chuv.ch), [jaime.barrancohernandez@hevs.ch](mailto:jaime.barrancohernandez@hevs.ch), [jaime.barrancohernandez@unil.ch](mailto:jaime.barrancohernandez@unil.ch)
- Benedetta Franceschiello — [benedetta.franceschiello@hevs.ch](mailto:benedetta.franceschiello@hevs.ch)  
- Meritxell Bach Cuadra — [meritxell.bachcuadra@unil.ch](mailto:meritxell.bachcuadra@unil.ch)

---

## 📄 License

This repository and associated software are distributed under the  
**“Software License Agreement for Academic Non-Commercial Research Purposes Only”**  
between **HES-SO Valais-Wallis** and **CHUV**.  

By downloading or using this software, you agree to the terms described in the [LICENSE](./LICENSE.txt) file.  
Usage is **strictly limited to academic and non-commercial research purposes**.  
For commercial or redistribution inquiries, please contact the licensors directly.

## 🏛️ Institutions

![Institutional logos](figs/logo-whitebg.png)

> © 2025 CIBM Center for Biomedical Imaging SP CHUV-UNIL, Lausanne, Switzerland and HES-SO University of Applied Sciences and Arts Western Switzerland, Sion, Switzerland.  
