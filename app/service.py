"""Gradio app."""

import gradio as gr

from utils import main


if __name__ == "__main__":

    css = ".output_image {height: 60rem !important; width: 100% !important;}"

    iface = gr.Interface(
        fn=main,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="Raw Image"),
            gr.Image(type="pil", label="Model Response"),
            gr.Dataframe(label="Model Predictions Metadata")
        ]
        css=css,
        title='NewsSonar Arquivo model.'
    )

    iface.launch(
        server_port=7860,
        server_name="0.0.0.0",
        debug=True
    )
