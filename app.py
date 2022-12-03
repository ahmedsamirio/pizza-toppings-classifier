from fastai.vision.all import *
import timm

import gradio as gr

def get_label(img_path):
    one_hot = df.query('image_name == @img_path.name').iloc[:, 1:].values[0]
    return one_hot

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Pizza Toppings Classifier"
description = "A pizza toppings classifier trained on the MIT pizza dataset with fastai. Created as a demo for Gradio and HuggingFace Spaces."
examples = ['onions_and_peppers.jpg', 'pepperoni_and_mushrooms.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=10)).launch(share=True)