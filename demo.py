import gradio as gr
from gradio_client import Client

gpt35_client = Client("http://0.0.0.0:8086/")
# moss_client = Client("http://0.0.0.0:8087/")
ch_llama_client = Client("http://0.0.0.0:8087")
chatglm_client = Client("http://0.0.0.0:8088/")
chatglm2_client = Client("http://0.0.0.0:8089/")

clients = [gpt35_client, ch_llama_client, chatglm_client, chatglm2_client]


def predict(text_input):
    jobs = []
    for idx, client in enumerate(clients):
        jobs.append(client.submit(text_input, 8192, 0.9, 0.95, fn_index=0))
    return tuple(eval(j.result()) for j in jobs)
 

def reset_states():
    jobs = []
    for client in clients:
        jobs.append(client.submit(fn_index=1))
    for job in jobs:
        job.result()
    return tuple([] for _ in jobs) 


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Multi-model ChatBot</h1>""")
    text_input = gr.Textbox(label="Input Text")
    with gr.Row():
        clear_button = gr.Button("清除历史对话")
        run_button = gr.Button("Chat!")
    with gr.Row():
        gpt35_bot = gr.Chatbot(label="ChatGPT 3.5")
        llama_bot = gr.Chatbot(label="Chinese LLaMA")
    with gr.Row():
        glm_bot = gr.Chatbot(label="ChatGLM")
        glm2_bot = gr.Chatbot(label="ChatGLM2")

    bots = [gpt35_bot, llama_bot, glm_bot, glm2_bot]

    example_inputs = [
        "你好呀！",
        "请问京东是一家怎样的公司集团？",
        "AI 领域的发展前景如何？"
    ]

    examples = gr.Examples(examples=example_inputs, inputs=text_input, outputs=[*bots], fn=predict, cache_examples=False)

    run_button.click(predict, [text_input], [*bots])
    clear_button.click(reset_states, [], [*bots])

demo.queue().launch(server_name="0.0.0.0", server_port=8090)
