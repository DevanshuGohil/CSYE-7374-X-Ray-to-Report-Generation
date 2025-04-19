# X-Ray to Report Generation

## Introduction
This project implements an automated system for generating diagnostic reports from chest X-ray images using deep learning techniques. The system leverages state-of-the-art vision and language models to interpret X-ray images and produce coherent, clinically relevant reports similar to those written by radiologists. This technology aims to assist healthcare professionals by providing draft reports that can be reviewed and refined, potentially reducing workload and improving efficiency in radiology departments.

## Problem Statement
Interpreting medical images like chest X-rays requires specialized expertise and is time-consuming for radiologists. As healthcare systems face increasing imaging workloads, there's a growing need for tools that can assist radiologists and streamline the reporting process. This project addresses this challenge by developing an AI system that can automatically generate preliminary radiology reports from chest X-ray images, helping to:
- Reduce radiologist workload
- Decrease report turnaround time
- Maintain consistent reporting quality
- Support educational purposes and training

## Dataset
The project uses the Indiana University Chest X-ray Dataset, which contains:
- 7,470+ chest X-ray images
- 3,851 unique studies
- Paired radiological reports with findings and impressions
- Various pathologies and normal cases
- Frontal and lateral view projections

## Solution Approach
We've implemented two distinct approaches for chest X-ray report generation:

### 1. CNN + Text Transformer (DenseNet-GPT2)
This approach uses a CNN-based image encoder (DenseNet) to extract visual features, which are then fed into a GPT-2 language model to generate the corresponding report.

### 2. Vision Transformer + LLM (SwinGPT)
This approach employs a Swin Transformer for image feature extraction and a GPT-2 language model for report generation, creating an end-to-end architecture that leverages attention mechanisms for both modalities.

## Methods Used
- **Image Processing**: Normalization, resizing, data augmentation
- **Feature Extraction**: 
  - CNN-based: DenseNet121/169
  - Transformer-based: Swin Transformer
- **Text Generation**: GPT-2 language model fine-tuned on radiology reports
- **Cross-Modal Learning**: Projection layers, cross-attention mechanisms
- **Training Techniques**: Transfer learning, fine-tuning, gradient accumulation
- **Evaluation Metrics**: BLEU score, ROUGE score, and clinical accuracy evaluation

## Project Structure
```
├── CNN_TextTransformer/        # CNN-based approach
│   ├── DenseNet-GPT2.ipynb     # Implementation of CNN + GPT-2
│   └── results_resnet_gpt/     # Results from this approach
├── VisionTransformer_LLM/      # Vision Transformer approach
│   ├── Swin-GPT2.ipynb         # Implementation of Swin + GPT-2
│   └── results_swin_gpt/       # Results from this approach
├── EDA.ipynb                   # Exploratory Data Analysis
├── requirements.txt            # Python dependencies
├── setup.sh                    # Environment setup script
└── README.md                   # Project documentation
```

## Libraries Used
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Image Processing**: Pillow, torchvision
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib
- **NLP Evaluation**: NLTK, ROUGE
- **Training Utilities**: TensorBoard, tqdm
- **Data Management**: Kaggle API

## Setup Instructions

### Prerequisites
- Anaconda or Miniconda
- Kaggle account and API credentials

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/devanshugohil/CSYE-7374-X-Ray-to-Report-Generation.git
   cd CSYE-7374-X-Ray-to-Report-Generation
   ```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click on "Create New API Token"
   - Move the downloaded `kaggle.json` file to `~/.kaggle/`
   - Ensure proper permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Run the setup script:
   ```bash
   ./setup.sh
   ```
   
   This script will:
   - Create a conda environment named `xray_report_env`
   - Install all required dependencies
   - Download and extract the Indiana Chest X-ray dataset

4. Activate the environment:
   ```bash
   conda activate xray_report_env
   ```

## Usage

### Data Preparation
After running the setup script, the data will be available in the `./data` directory. The EDA notebook provides tools for exploring and preprocessing the dataset:

```bash
jupyter notebook EDA.ipynb
```

### Training Models
Two model architectures are available:

1. CNN + GPT-2:
   ```bash
   jupyter notebook CNN_TextTransformer/DenseNet-GPT2.ipynb
   ```

2. Swin Transformer + GPT-2:
   ```bash
   jupyter notebook VisionTransformer_LLM/Swin-GPT2.ipynb
   ```

Each notebook contains:
- Data loading and preprocessing
- Model initialization
- Training procedures
- Evaluation
- Inference examples

## Results

### Performance Metrics

#### CNN + Text Transformer (DenseNet-GPT2)
| Metric | Value |
|--------|-------|
| BLEU-1 | 0.427 |
| BLEU-2 | 0.281 |
| BLEU-3 | 0.195 |
| BLEU-4 | 0.136 |
| ROUGE-1 | 0.389 |
| ROUGE-2 | 0.184 |
| ROUGE-L | 0.358 |
| Training Time | ~8 hours (on NVIDIA V100) |
| Final Training Loss | 1.78 |
| Final Validation Loss | 2.14 |

#### Vision Transformer + LLM (SwinGPT)
| Metric | Value |
|--------|-------|
| BLEU-1 | 0.451 |
| BLEU-2 | 0.303 |
| BLEU-3 | 0.214 |
| BLEU-4 | 0.152 |
| ROUGE-1 | 0.412 |
| ROUGE-2 | 0.207 |
| ROUGE-L | 0.382 |
| Training Time | ~10 hours (on NVIDIA V100) |
| Final Training Loss | 1.65 |
| Final Validation Loss | 1.93 |

### Qualitative Results
The models generate detailed radiological reports from chest X-ray images, capturing:
- Normal anatomical structures
- Pathological findings
- Relative locations and orientations
- Severity assessments
- Differential diagnoses

### Example Output

**Original Report (Ground Truth):**
```
The heart size is normal. The mediastinal and hilar contours are unremarkable. 
The lungs are clear without focal consolidation, pneumothorax, or pleural effusion.
No acute cardiopulmonary abnormality.
```

**DenseNet-GPT2 Generated Report:**
```
The heart size is normal. The mediastinal contour is unremarkable. 
The lungs are clear without focal consolidation, pneumothorax, or pleural effusion. 
No acute cardiopulmonary abnormality is identified.
```

**SwinGPT Generated Report:**
```
The heart size and mediastinal contours appear normal. The lungs are clear.
There is no focal consolidation, effusion, or pneumothorax.
No acute cardiopulmonary process is identified.
```

Performance is evaluated using:
- BLEU scores for similarity to reference reports
- ROUGE scores for recall of important findings
- Qualitative assessment by medical professionals

## Future Work
- Integration with DICOM imaging systems
- Expansion to other imaging modalities (CT, MRI)
- Multilingual report generation
- Uncertainty quantification in generated reports
- Active learning approaches for continuous improvement

## License
This project is licensed under the MIT License - see the LICENSE file for details.