import os
os.system("python setup.py build develop --user")

import gradio as gr

from app_util import ContextDetDemo

header = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
Contextual Object Detection with Multimodal Large Language Models
</h1>
</div>
'''

abstract = '''
🤗 This is the official Gradio demo for <b>Contextual Object Detection with Multimodal Large Language Models</b>.

🆒 Our goal is to promote object detection with better `context understanding` and enable `interactive feedback`
through `human language vocabulary`, all made possible by using multimodal large language models!

🤝 This demo is still under construction. Your comments or suggestions are welcome!

⚡ For faster inference without waiting in the queue, you may duplicate the space and use the GPU setting:
<a href="https://huggingface.co/spaces/yuhangzang/ContextDet-Demo?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
<p/>
'''

footer = r'''
🦁 **Github Repo**
We would be grateful if you consider star our <a href="https://github.com/yuhangzang/ContextDET">github repo</a>

📝 **Citation**
We would be grateful if you consider citing our work if you find it useful:
```bibtex
@article{zang2023contextual,
  author = {Zang, Yuhang and Li, Wei and Han, Jun, and Zhou, Kaiyang and Loy, Chen Change},
  title = {Contextual Object Detection with Multimodal Large Language Models},
  journal = {arXiv preprint arXiv:2305.18279},
  year = {2023}
}
```

📋 **License**
This project is licensed under
<a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">S-Lab License 1.0</a>.
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**
If you have any questions, please feel free to contact Yuhang Zang <b>(zang0012@ntu.edu.sg)</b>.
'''

css = '''
h1#title {
  text-align: center;
}
'''

cloze_samples = [
    ["main_4.jpg", "A teacher is helping a <mask> with her homework at desk."],
    ["main_5.jpg", "A man crossing a busy <mask> with his <mask> up."],
]


captioning_samples = [
    ["main_1.jpg"],
    ["main_2.jpg"],
    ["main_4.jpg"],
    ["main_6.jpeg"],
]

qa_samples = [
    ["main_5.jpg", "What is his career?"],
    ["main_6.jpeg", "What are they doing?"],
]

contextdet_model = ContextDetDemo('./ckpt.pth')


def inference_fn_select(image_input, text_input, task_button, history=[]):
    return contextdet_model.forward(image_input, text_input, task_button, history)


def set_cloze_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(example[1]), 'Cloze Test'


def set_captioning_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(''), 'Captioning'


def set_qa_samples(example: list) -> dict:
    return gr.Image.update(example[0]), gr.Textbox.update(example[1]), 'Question Answering'


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(header)
    gr.Markdown(abstract)
    state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_input = gr.Image(type="pil", interactive=True, label="Upload an image 📁").style(height=250)
        with gr.Column(scale=0.5, min_width=500):
            chat_input = gr.Textbox(label="Type your text prompt ⤵️")
            task_button = gr.Radio(label="Contextual Task type", interactive=True,
                                   choices=['Cloze Test', 'Captioning', 'Question Answering'],
                                   value='Cloze Test')
            with gr.Row():
                submit_button = gr.Button(value="🏃 Run", interactive=True, variant="primary")
                clear_button = gr.Button(value="🔄 Clear", interactive=True)

    with gr.Row():
        with gr.Column(scale=0.5, min_width=500):
            image_output = gr.Image(type='pil', interactive=False, label="Detection output")
        with gr.Column(scale=0.5, min_width=500):
            chat_output = gr.Chatbot(label="Text output").style(height=300)

    with gr.Row():
        with gr.Column(scale=0.33, min_width=330):
            cloze_examples = gr.Dataset(
                label='Contextual Cloze Test Examples',
                components=[image_input, chat_input],
                samples=cloze_samples,
            )
        with gr.Column(scale=0.33, min_width=330):
            qa_examples = gr.Dataset(
                label='Contextual Question Answering Examples',
                components=[image_input, chat_input],
                samples=qa_samples,
            )
        with gr.Column(scale=0.33, min_width=330):
            captioning_examples = gr.Dataset(
                label='Contextual Captioning Examples',
                components=[image_input, ],
                samples=captioning_samples,
            )

    submit_button.click(
        inference_fn_select,
        [image_input, chat_input, task_button, state],
        [image_output, chat_output, state],
    )
    clear_button.click(
        lambda: (None, None, "", [], [], 'Question Answering'),
        [],
        [image_input, image_output, chat_input, chat_output, state, task_button],
        queue=False,
    )
    image_input.change(
        lambda: (None, "", []),
        [],
        [image_output, chat_output, state],
        queue=False,
    )
    cloze_examples.click(
        fn=set_cloze_samples,
        inputs=[cloze_examples],
        outputs=[image_input, chat_input, task_button],
    )
    captioning_examples.click(
        fn=set_captioning_samples,
        inputs=[captioning_examples],
        outputs=[image_input, chat_input, task_button],
    )
    qa_examples.click(
        fn=set_qa_samples,
        inputs=[qa_examples],
        outputs=[image_input, chat_input, task_button],
    )

    gr.Markdown(footer)

demo.launch(enable_queue=True, share=False)
# demo.launch(enable_queue=True, share=True)