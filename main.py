from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
import streamlit as st
# Choose any model available at https://health.petals.dev
model_name = "petals-team/StableBeluga2"  # This one is fine-tuned Llama 2 (70B)

# Connect to a distributed network hosting model layers
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name,device_map='auto')

# Run the model as if it were on your computer
system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"
p=st.chat_input("Say something")
if p:
    message = p
    prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    outputs = model.generate(**inputs, max_new_tokens=100)
    out=tokenizer.decode(outputs[0],  skip_special_tokens=True)
    out=out.partition('### Assistant:\n')[-1]
    st.chat_message('user').write(p)
    st.chat_message('assistant').write(out)