import gradio as gr
from gesture_recognition_tools import main


tools = main()
model = tools.model
resultText = ""


def run(im):
    global resultText
    result = model.process_frame(im)
    if result is not None:
        if result["label"] is not None:
            resultText += f"{result['label']} "
            return resultText

    return resultText


demo = gr.Interface(
    run,
    gr.Image(source="webcam", streaming=True, shape=(580, 700), type="numpy"),
    "text",
    live=True,
)
demo.launch()
