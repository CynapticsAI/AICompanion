from transformers import pipeline, set_seed

# set a random seed for reproducibility
set_seed(42)

# instantiate the pipeline with the personaGPT model
generator = pipeline('text-generation', 
                     model='af1tang/personaGPT', 
                     tokenizer='af1tang/personaGPT')

# generate text using the personaGPT model
prompt = "Hi, my name is Sarah and I love to play tennis. What do you like to do?"
generated_text = generator(prompt, max_length=100)[0]['generated_text']

# print the generated text
print(generated_text)