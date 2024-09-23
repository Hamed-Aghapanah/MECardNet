
# MECardNet: A Novel Multi-Scale Convolutional Ensemble Model with Adaptive Deep Supervision for Precise Cardiac MRI Segmentation

## Overview
MECardNet is an innovative multi-class segmentation framework for Cardiac MRI data. This model represents a major advancement in the field by leveraging a Multi-scale Convolutional Mixture of Experts (MCME) ensemble technique and Adaptive Deep Supervision for highly accurate segmentation of left ventricle, right ventricle, and myocardium.

![Cardiac Segmentation Overview](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide16.PNG)

## Graphical Abstract
The graphical abstract below illustrates the architecture and flow of the MECardNet model for segmenting Cardiac MRI data.

![Graphical Abstract](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide19.PNG)

## Architecture
The overall architecture of MECardNet is designed based on U-Net, with several key components added for feature enhancement.

![MECardNet Architecture](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide1.PNG)

## Multi-scale Convolutional Mixture of Experts
The MCME Block combines features from multiple scales to improve the representation and accuracy of segmentation.

![MCME Block](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide2.PNG)

## Features
- **Multi-scale Convolutional Mixture of Experts (MCME)**: Enhances representation learning within the U-Net architecture by adaptively combining layers for better data modeling.
- **Adaptive Deep Supervision (ADS)**: Introduces supervisory signals at multiple layers to improve the robustness of the model and reduce errors.
- **EfficientNetV2L Backbone**: Facilitates efficient hierarchical feature extraction.
- **Cross-Additive Attention Mechanism**: Improves the model's ability to focus on relevant information across different scales within the data.
- **Specialized Loss Function**: Utilizes Dice and curvature loss functions for optimal performance in segmentation accuracy.

## Performance
MECardNet surpasses state-of-the-art methods, achieving:
- **Dice Similarity Coefficient (DSC)**: 96.1% ± 0.4%
- **Jaccard Coefficient**: 92.2% ± 0.4%
- **Hausdorff Distance**: 1.7 ± 0.1
- **Mean Absolute Distance**: 1.6 ± 0.1

![Performance Comparison](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide5.PNG)

The model has been validated on the ACDC dataset as well as the M&Ms-2 and a local dataset, showcasing its robust generalization capabilities.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Hamed-Aghapanah/MECardNet.git
    ```
2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the script to train MECardNet:
    ```bash
    python train_mecardnet.py --dataset ACDC --epochs 50
    ```
2. To evaluate the model:
    ```bash
    python evaluate_mecardnet.py --dataset ACDC
    ```

## Datasets
- **ACDC 2017**: Automated Cardiac Diagnosis Challenge dataset.
- **M&Ms-2**: Multi-Disease, Multi-View, and Multi-Center cardiac MRI dataset.
- **Rajaie CMRI Dataset**: Collected from Shahid Rajaie Cardiovascular Medical and Research Center, Tehran, Iran.

![Datasets Comparison](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide20.PNG)

## Post Processing
Post-processing techniques help refine the segmentation results to ensure the anatomical accuracy of the predictions.

![Post Processing Steps](https://github.com/Hamed-Aghapanah/MECardNet/blob/main/fig/croped_Slide3.PNG)

## Results
The performance of MECardNet on different datasets can be found in the `results/` directory. Example segmentation outputs and detailed performance metrics are also available.

## Citation
If you use this model or its components in your research, please cite our work:
```
@article{Aghapanah2024,
  title={MECardNet: A Novel Multi-Scale Convolutional Ensemble Model with Adaptive Deep Supervision for Precise Cardiac MRI Segmentation},
  author={Hamed Aghapanah, Reza Rasti, Faezeh Tabesh, Hamidreza Pouraliakbar, Hamid Sanei, Saeed Kermani},
  journal={Journal of Medical Imaging and Analysis},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
