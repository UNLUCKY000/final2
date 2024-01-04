import streamlit as st 
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

st.title('Test')

if 'cache' not in st.session_state:
    st.session_state.cache = {'TEXT_PIPE' : pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto"),
                            'AUDIO_PIPE': pipeline("text-to-speech", "microsoft/speecht5_tts"),
                            'EMBED':load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")}

if st.button('Generate Story'): 
    st.write('TEST')

    pipe = st.session_state.cache['TEXT_PIPE']

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {"role": "user", "content": "Write a story. ..."},
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=10, do_sample=True, temperature=0.7, top_k=30, top_p=0.90)
    text = outputs[0]["generated_text"]

    # text = outputs = '<|assistant|> here we go, if u wanna try, lets do it'
    start = '<|assistant|>'
    text = text[text.index(start)+len(start):]
    st.write(text)

    embeddings_dataset = st.session_state.cache['EMBED']
    synthesiser = st.session_state.cache['AUDIO_PIPE']
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    speech = synthesiser(text, forward_params={"speaker_embeddings": speaker_embedding})
    # st.write(speech)
    st.audio(speech['audio'], sample_rate = speech['sampling_rate'])

