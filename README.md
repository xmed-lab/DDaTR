<div align='center'>

<h2><a href="https://ieeexplore.ieee.org/document/11087655">DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation</a></h2>

[Shanshan Song](https://scholar.google.com.hk/citations?user=EoNWyTcAAAAJ&hl=zh-CN), [Tang Hui](https://scholar.google.com/citations?user=eqVvhiQAAAAJ&hl=zh-CN), [Honglong Yang](https://scholar.google.com/citations?user=3BPUjoQAAAAJ&hl=zh-CN), [Xiaomeng Li](https://scholar.google.com.hk/citations?hl=zh-CN&user=uVTzPpoAAAAJ)
 
Hong Kong University of Science and Technology (HKUST)

</div>


## 🔨 Installation

Clone this repository and install the required packages:

```shell
git clone https://github.com/xmed-lab/DDaTR.git
cd DDaTR

conda create -n ddatr python=3.10
conda activate ddatr
pip install -r requirements.txt
```


## 🍹 Preparation

### Data Acquisition

* Images: the images can be downloaded from [MIMIC CXR](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) and [IU-XRay](https://github.com/zhjohnchan/R2Gen)

* Annotation: 
    * [MIMIC CXR](https://drive.google.com/file/d/1UZFGA8FXuYfnN23eV3ksxoBgRwWKfPZR/view?usp=sharing). Put at ./data/mimic_cxr
    * [Longitudinal-MIMIC](https://drive.google.com/file/d/1FjMcvUQqDIyCgkd4ySBIecCh4eiG_o-V/view?usp=sharing). Put at ./data/mimic_cxr
    * [ReXrank test](https://drive.google.com/file/d/1erDVnUJ-xJ84PkrcFp_hHj_m5EP_DhBs/view?usp=sharing)
    * IU-XRay: please download from [PromptMRG](https://github.com/jhb86253817/PromptMRG/) 

### Pre-trained Weight Downloading

* Checkpoint:
    * [chexbert](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9)

## 🍻 Quick Start for Training & Evaluation
After all the above preparation steps, you can train DDaTR with the following command: 
```shell
# For MIMIC-CXR and Longitudinal-MIMIC
bash train_mimic_cxr.sh
```

You can directly download our trained model from [DDaTR](https://drive.google.com/file/d/1zd4rEtDKbzEx0bGAWu4CtuQvRsNJaEMg/view?usp=sharing)

For testing, you can use following command: 
```shell
# For MIMIC-CXR and Longitudinal-MIMIC
bash test_mimic_cxr.sh

# For ReXrank
bash test_rexrank.sh

# For IU-XRay
bash test_iu_xray.sh
```

## 💙 Acknowledgement

DDaTR is built upon the awesome [PromptMRG](https://github.com/jhb86253817/PromptMRG/), [LAVT](https://github.com/yz93/LAVT-RIS), [LDCNet](https://github.com/huiyu8794/LDCNet).

## 📄 Citation

Paper link: [DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation](https://ieeexplore.ieee.org/document/11087655).

If you use this work in your research, please cite:

```bibtex
@ARTICLE{DDaTR,
  author={Song, Shanshan and Tang, Hui and Yang, Honglong and Li, Xiaomeng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={DDaTR: Dynamic Difference-aware Temporal Residual Network for Longitudinal Radiology Report Generation}, 
  year={2025},
  pages={1-12},
  keywords={Radiology report generation;Longitudinal radiology report generation;Dynamic difference-awareness;Longitudinal multimodal encoder},
  doi={10.1109/TMI.2025.3591364}}

```

## 📧 Contact

For questions and issues, please use the GitHub issue tracker or contact [ssongan@connect.ust.hk]. 
