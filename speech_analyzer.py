from transformers import pipeline
import gradio as gr

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# IBM watsonx credentials
my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

params = {
    GenParams.MAX_NEW_TOKENS: 800,
    GenParams.TEMPERATURE: 0.1,
}

LLAMA2_model = Model(
    model_id="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
    credentials=my_credentials,
    params=params,
    project_id="skills-network",
)

llm = WatsonxLLM(LLAMA2_model)

# Prompt template
template = """
<s><<sys>>
List the key points with details from the context:
[INST] The context : {context} [/INST]
<</sys>>
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=template
)

prompt_to_LLAMA2 = LLMChain(
    llm=llm,
    prompt=prompt
)

# Speech-to-text + summarization
def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    transcript_txt = pipe(audio_file, batch_size=8)["text"]

    result = prompt_to_LLAMA2.run(transcript_txt)

    return result


# Gradio interface
audio_input = gr.Audio(
    sources="upload",
    type="filepath"
)

output_text = gr.Textbox(label="Summary")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=output_text,
    title="Audio Transcription App",
    description="Upload an audio file to transcribe and summarize."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)