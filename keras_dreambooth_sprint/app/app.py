from huggingface_hub import from_pretrained_keras
from keras_cv import models
import gradio as gr
import tensorflow as tf

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# load keras model
resolution = 512
dreambooth_model = models.StableDiffusion(
        img_width=resolution, img_height=resolution, jit_compile=True, 
    )
loaded_diffusion_model = from_pretrained_keras("tgohblio/dreambooth_hogwarts_legacy")
dreambooth_model._diffusion_model = loaded_diffusion_model


# generate images
def generate_images(prompt: str, negative_prompt: str, num_imgs_to_gen: int, inference_steps: int, guidance_scale: float):
    output_images = dreambooth_model.text_to_image(
        prompt,
        negative_prompt=negative_prompt,
        batch_size=num_imgs_to_gen,
        num_steps=inference_steps,
        unconditional_guidance_scale=guidance_scale,
    )
    return output_images
    
# Define the UI
with gr.Blocks() as demo:
    gr.HTML("<h2 style=\"font-size: 2rem; font-weight: 700; text-align: center;\">Keras Dreambooth - Hogwarts Legacy Demo</h2>")
    gr.HTML("<h3 style=\"font-size: 2rem; font-weight: 700; text-align: left;\">This model has been fine-tuned to learn the concept of Hogwarts Legacy student characters. \
        To use this demo, you should have append your prompt with string \'hogwarts [legacy] student\'</h3>")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Positive Prompt", value="a digital art of hogwarts [legacy] student in a forest")
            negative_prompt = gr.Textbox(label="Negative Prompt", value="bad anatomy, soft blurry")
            samples = gr.Slider(label="Number of Images", minimum=1, maximum=4, value=1, step=1)
            inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=50, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=10, value=7.5, step=0.1)
            run = gr.Button(value="Run")
        with gr.Column():
            gallery = gr.Gallery(label="Outputs").style(grid=(1,2))

    run.click(fn=generate_images, inputs=[prompt, negative_prompt, samples, inference_steps, guidance_scale], outputs=gallery)
    
    gr.Examples([["realistic painting of a hogwarts [legacy] student riding a horse, high quality, 8k", "bad, ugly, malformed, deformed, out of frame, blurry, cropped, noisy", 4, 50, 7.5]],
                [prompt, negative_prompt, samples, inference_steps, guidance_scale], gallery, generate_images, cache_examples=True)
    gr.Markdown('Demo created by [Terrence Goh](https://huggingface.co/tgohblio/)')

demo.queue(concurrency_count=3)
demo.launch()    
# Style 2
# --instance_prompt="classic animation style" \
# --class_prompt="illustration style" \