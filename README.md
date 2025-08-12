# AI Image Upscaling and Inpainting

A comprehensive AI-powered solution for image upscaling and inpainting using state-of-the-art deep learning models.

<img width="302" height="226" alt="Image" src="https://github.com/user-attachments/assets/acf5ecc8-b74f-4cd4-9cdd-4eaaa481689c" />
<img width="2304" height="1664" alt="Image" src="https://github.com/user-attachments/assets/20d4357b-a031-4cbf-b3b0-99b660161c2e" />
<img width="287" height="176" alt="Image" src="https://github.com/user-attachments/assets/426bdda4-f675-465e-b876-e0497cb884dd" />
<img width="2560" height="1536" alt="Image" src="https://github.com/user-attachments/assets/c4fa9403-942c-470e-9cf4-9beebbeb74c2" />
<img width="320" height="180" alt="Image" src="https://github.com/user-attachments/assets/7d438858-8590-40ef-b9bb-1bb75e56bc50" />
<img width="2688" height="1536" alt="Image" src="https://github.com/user-attachments/assets/d404df15-8460-4db1-b1e6-9e765f960670" />

---

<img width="2880" height="2880" alt="Image" src="https://github.com/user-attachments/assets/e647b315-5a56-4c3f-8406-bab81ab1649f" />
<img width="2880" height="2880" alt="Image" src="https://github.com/user-attachments/assets/ad455fcf-bdd0-4b8d-9a9d-adca1a665918" />

## Features

- **Image Upscaling**: Upscale low-resolution images to high resolution using ESRGAN and Real-ESRGAN models
- **Image Inpainting**: Fill in missing or damaged parts of images using diffusion models
- **Multiple AI Models**: Support for various pre-trained models
- **User-Friendly Interface**: Web-based UI using Streamlit and Gradio
- **Batch Processing**: Process multiple images at once
- **Quality Control**: Adjustable parameters for optimal results

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```

### Web Interface (Gradio)
```bash
python gradio_app.py
```

### Command Line
```bash
python main.py --input image.jpg --output upscaled.jpg --mode upscale --scale 4
```

## Models

### Upscaling Models
- **ESRGAN**: Enhanced Super-Resolution Generative Adversarial Networks
- **Real-ESRGAN**: Real-ESRGAN for general image restoration
- **EDSR**: Enhanced Deep Residual Networks for Super Resolution

### Inpainting Models
- **Stable Diffusion Inpainting**: State-of-the-art inpainting using diffusion models
- **LaMa**: Large Mask Inpainting for irregular masks

## Project Structure

```
├── models/           # AI model implementations
├── utils/            # Utility functions
├── data/             # Sample data and models
├── app.py            # Streamlit web application
├── gradio_app.py     # Gradio web application
├── main.py           # Command line interface
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## License

MIT License - see LICENSE file for details. 
