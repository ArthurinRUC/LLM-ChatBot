import os
import json
import base64
import gradio as gr
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int)
args = parser.parse_args()
port = args.port

URL = "http://easyalgo.jd.com/openapi/chatgpt/callGpt"
USERNAME = os.environ["GPT35_USERNAME"]
PASSWORD = os.environ["GPT35_PASSWORD"]


def gen_authorization(username, password):
    s = username + ":" + password
    b = s.encode("utf-8")
    return str(base64.b64encode(b), "utf-8");


def gen_headers(authorization):
    auth = f"Basic {authorization}"
    headers = {
        "Authorization": auth
    }
    return headers


def gen_body_str(user_message, chat=False, history_message=None, ext=None):
    body = {
        "user_message": user_message,
    }

    if chat:
        body["history_message"] = history_message

    if ext:
        body.update(**ext)

    return json.dumps(body, ensure_ascii=False).encode("utf-8")


chatbot = []


def predict(input, max_length, top_p, temperature, history):
    global chatbot
    user_message = {
        "role": "user",
	"content": f"{input}"
    }
    body = gen_body_str([user_message], chat=True, history_message=history)
    headers = gen_headers(gen_authorization(USERNAME, PASSWORD))
    r = requests.post(URL, data=body, headers=headers)
    content = json.loads(r.content)

    try:
        response_message = content["data"]["choices"][0]["message"]
    except KeyError:
        response_message = {"content": ""}

    chatbot.append((input, response_message["content"].strip()))
    history.extend([user_message, response_message])

    return str(chatbot), history


def reset_state():
    global chatbot
    chatbot = []
    return []


with gr.Blocks() as demo:
    run = gr.Button("Submit")
    clear = gr.Button("Clear")
    msg = gr.Textbox(label="input")
    text = gr.Textbox(label="chatbot")
    max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
    temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
    
    text = gr.Textbox()
    history = gr.State([])

    run.click(predict, [msg, max_length, top_p, temperature, history], [text, history])
    clear.click(reset_state, [], [history])


demo.queue().launch(server_name="0.0.0.0", server_port=port)
