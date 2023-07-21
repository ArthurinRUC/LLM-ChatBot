import argparse
import gradio as gr
import mdtex2html
import transformers
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig

parser = argparse.ArgumentParser(description="ChatGLM2 Model Conversation")
parser.add_argument("--port", type=int)
parser.add_argument("--model_dir", type=str, help="pretrained model dir")
parser.add_argument("--peft_model_dir", type=str, help="peft model dir")
args = parser.parse_args()
port = args.port
model_dir = args.model_dir
peft_model_dir = args.peft_model_dir

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()

if peft_model_dir is not None:
    model = PeftModel.from_pretrained(model, peft_model_dir)

model = model.eval()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess

chatbot = []

def reset_state():
    global chatbot
    chatbot = []
    return [], None


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text




def predict(input, max_length, top_p, temperature, history, past_key_values):
    global chatbot
    chatbot.append((parse_text(input), ""))
    response, history = model.chat(tokenizer, input, history, max_length=max_length, top_p=top_p, temperature=temperature)
    chatbot[-1] = (parse_text(input), parse_text(response))

    return str(chatbot), history, past_key_values


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM2-6B</h1>""")

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...").style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
    text = gr.Textbox()
    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, max_length, top_p, temperature, history, past_key_values],
                    [text, history, past_key_values], show_progress=True)
    emptyBtn.click(reset_state, outputs=[history, past_key_values], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", server_port=port)
