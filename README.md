# Practical No. 05: Image Generation Using Pretrained Stable Diffusion

**Student Name:** Gayatri Pandharinath Gaikwad  
**Roll Number:** 202502110002  
**Class:** TY – CSE (AIML)  
**GitHub Repository:** https://github.com/GaikwadGayatri16/Image-Generation-using-Pretrained-Stable-Diffusion

---

## 1. Project Overview

This practical implements a pretrained Stable Diffusion model using the Hugging Face Diffusers library to generate images from natural-language text prompts.

The project explores how prompt engineering, negative prompts, inference steps, guidance scale, sampling schedulers, and random seeds influence the generated images. It also demonstrates the internal denoising process and evaluates outputs using generation time, CLIP-based image-text similarity, and mean squared error (MSE).

The pretrained model used is **Stable Diffusion v1.5**. The model is used for inference without training it from scratch.

## 2. Aim

To implement a pretrained Stable Diffusion model using Hugging Face Diffusers and analyse the impact of inference steps, guidance scale, prompt variations, negative prompts, and different sampling methods on generated image quality.

## 3. Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- Accelerate
- Safetensors
- NumPy
- Pandas
- Matplotlib
- Pillow (PIL)
- Jupyter Notebook / Google Colab

## 4. Stable Diffusion Architecture

Stable Diffusion is a latent diffusion model that generates images by gradually removing noise from a latent representation.

The major components are:

**1. Tokenizer:** Converts the input text prompt into tokens.

**2. Text Encoder:** Converts the tokens into embeddings that represent the meaning of the prompt.

**3. U-Net:** Predicts the noise present in the latent representation at each denoising timestep.

**4. Scheduler:** Controls the denoising timesteps and updates the latent representation using the predicted noise.

**5. Variational Autoencoder (VAE):** Decodes the final latent representation into a viewable image.

**6. Classifier-Free Guidance:** Controls how strongly the generated image follows the text prompt by combining conditional and unconditional predictions.

The model performs denoising in latent space rather than directly processing the full image at every step. This reduces computational requirements compared with pixel-space diffusion.

## 5. Installation and Setup

Google Colab with a GPU runtime is recommended for faster execution.

Install the required libraries using:

```python
!pip install -q -U diffusers transformers accelerate safetensors scipy matplotlib pandas
```

### Execution Steps

1. Open the Jupyter Notebook in Google Colab or a local Jupyter environment.
2. Select a GPU runtime if available.
3. Install the required libraries.
4. Execute the notebook cells sequentially.
5. Load the pretrained Stable Diffusion model.
6. Generate images using the provided prompts.
7. Run the parameter experiments and inspect the resulting images, tables, and plots.

The notebook uses CUDA when available and otherwise falls back to CPU. CPU execution may be significantly slower. The first run requires downloading the pretrained model files.

## 6. Experiments Performed

### 6.1 Basic Text-to-Image Generation

A text prompt describing a lighthouse at sunset is used to generate an image. The baseline experiment demonstrates how a pretrained Stable Diffusion pipeline converts a text description into an image.

### 6.2 Manual Denoising Process

The notebook demonstrates the internal process of Stable Diffusion:

1. Convert the text prompt into embeddings.
2. Initialize a latent representation with random noise.
3. Use the U-Net to predict noise at each timestep.
4. Apply classifier-free guidance.
5. Update the latent using the scheduler.
6. Decode the final latent using the VAE.

Intermediate outputs illustrate how the latent representation develops during denoising.

### 6.3 Prompt Engineering

Different prompt variations are tested, including basic descriptions, detailed scene descriptions, lighting conditions, and artistic styles such as oil painting, watercolor, and cyberpunk.

The experiment demonstrates how prompt wording influences image composition, appearance, and style.

### 6.4 Negative Prompt Comparison

Positive prompts are tested with and without negative prompts.

Negative prompts include unwanted features such as blur, low quality, distorted anatomy, extra fingers, watermarks, and unwanted text.

The experiment evaluates whether negative prompts help reduce undesirable visual features.

### 6.5 Effect of Inference Steps

The following inference-step values are tested:

- 5 steps
- 10 steps
- 20 steps
- 30 steps
- 50 steps

The prompt, random seed, guidance scale, and scheduler are kept constant.

Generation time and CLIP-based image-text similarity are recorded to study the trade-off between computational cost and output characteristics.

### 6.6 Effect of Guidance Scale

The following guidance scales are tested:

- 1
- 3
- 5
- 7.5
- 10
- 15
- 20

The experiment examines how the strength of prompt conditioning affects image-text alignment and visual output.

### 6.7 Comparison of Sampling Schedulers

The following schedulers are compared:

- PNDM
- DDIM
- Euler
- Euler Ancestral
- LMS
- DPM++ 2M
- UniPC

The schedulers are evaluated using 20 and 8 inference steps. Generation time and MSE against a selected reference image are recorded.

### 6.8 Combined Parameter Analysis

A combined experiment explores different inference-step counts and guidance scales to understand how these parameters interact.

### 6.9 Random Seed Variation

Different random seeds, including 1, 42, 123, and 2024, are tested to demonstrate how initial random noise affects the generated image.

Keeping the seed constant during controlled experiments helps make comparisons more meaningful.

## 7. Results and Output Analysis

The following observations were recorded in the notebook's saved outputs.

### 7.1 Inference Steps

Generation time increased from approximately 1.6 seconds at 5 steps to 13.0 seconds at 50 steps.

The highest recorded CLIP score in the tested inference-step experiment was 35.52 at 10 steps.

This indicates that increasing inference steps increases computational cost, but a higher step count does not necessarily produce a higher CLIP score.

### 7.2 Guidance Scale

The highest recorded CLIP score in the guidance-scale experiment was 35.28 at a guidance scale of 15.

Guidance scale influences how strongly the model follows the prompt. Very high guidance may introduce oversaturation, harsh contrast, or other visual artefacts.

The highest CLIP score does not automatically identify the visually best image.

### 7.3 Scheduler Comparison

At 20 inference steps, UniPC recorded the lowest MSE against the selected reference image, approximately 0.0030, followed by DPM++ 2M at approximately 0.0033.

At 8 steps, DPM++ 2M recorded the lowest MSE, approximately 0.0167.

In the saved timing results, Euler was the fastest scheduler at 20 steps, taking approximately 4.77 seconds. At 8 steps, Euler Ancestral was the fastest, taking approximately 2.05 seconds.

These results are specific to the tested configuration and runtime.

### 7.4 Interpretation of Evaluation Metrics

**CLIP similarity:** Measures the alignment between an image and a text description. It does not fully measure aesthetics, realism, or the presence of visual defects.

**Mean Squared Error (MSE):** Measures pixel-level differences between an image and a reference image. A lower MSE means the image is numerically closer to that reference, but it does not necessarily mean that the image is visually superior.

**Generation time:** Measures the time required to generate an image under the tested execution conditions. It depends on hardware, runtime configuration, and software versions.

Generated image grids should therefore be inspected alongside the numerical results.

## 8. Output Files

The notebook generates images, plots, and CSV tables in an `outputs/` directory when the corresponding cells execute successfully.

Expected output files include:

- `01_basic.png` — basic generated image
- `02_denoising_progress.png` — denoising progression
- `02_manual_loop.png` — manual denoising output
- `03_prompt_variants.png` — prompt variation comparison
- `04_negative_prompt.png` — negative prompt comparison
- `05_steps.png` — inference-step comparison
- `05_steps_plot.png` — inference-step analysis plot
- `06_guidance.png` — guidance-scale comparison
- `06_guidance_plot.png` — guidance-scale analysis plot
- `07_schedulers_20steps.png` — scheduler comparison at 20 steps
- `07_schedulers_8steps.png` — scheduler comparison at 8 steps
- `08_steps_x_guidance.png` — combined parameter experiment
- `09_seeds.png` — random-seed comparison

Expected CSV result tables include:

- `table_prompts.csv`
- `table_steps.csv`
- `table_guidance.csv`
- `table_schedulers.csv`

These files are created only if the corresponding notebook cells run successfully.

## 9. Key Learning Outcomes

This practical provides experience in:

- Implementing pretrained Stable Diffusion using Hugging Face Diffusers.
- Understanding text conditioning and latent diffusion.
- Explaining the roles of the text encoder, U-Net, scheduler, and VAE.
- Generating images using text prompts.
- Applying prompt engineering and negative prompts.
- Analysing the effects of inference steps and guidance scale.
- Comparing different sampling schedulers.
- Evaluating generated images using quantitative metrics.
- Understanding the importance of fixed random seeds in controlled experiments.

## 10. Limitations

1. The experiments use a limited number of prompts and parameter combinations.
2. CLIP similarity does not completely represent visual quality.
3. MSE against a generated reference does not provide a universal quality ranking.
4. Stable Diffusion may generate distorted details, incorrect text, or other artefacts.
5. Generation time varies with hardware, GPU availability, and software configuration.
6. Results may differ when the random seed or model configuration changes.

## 11. Conclusion

This practical demonstrates text-to-image generation using pretrained Stable Diffusion v1.5 and the Hugging Face Diffusers library.

The experiments show that prompt design, negative prompts, inference steps, guidance scale, scheduler selection, and random seed influence the generated output. Increasing inference steps generally increases computation time, while guidance scale affects prompt conditioning. Different schedulers may provide different trade-offs between execution time and similarity to a selected reference.

The practical also demonstrates why quantitative metrics should be interpreted alongside visual inspection. No single metric is sufficient to determine overall image quality.

Overall, the implementation provides an understanding of latent diffusion, iterative denoising, prompt conditioning, and parameter evaluation in text-to-image generation.

## 12. GitHub Repository

**Repository Link:** https://github.com/GaikwadGayatri16/Image-Generation-using-Pretrained-Stable-Diffusion
Replace the placeholder with your actual GitHub repository URL before submission.

The repository should contain the Jupyter Notebook, this README file, and any selected result images or plots required for demonstration. Avoid uploading large pretrained model weights or unnecessary temporary files.
