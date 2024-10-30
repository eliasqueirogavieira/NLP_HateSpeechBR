import torch
from transformers import pipeline
import pandas as pd
import re

# Define the model ID
model_id = "meta-llama/Llama-3.2-3B"

# Initialize the text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,  # Use appropriate dtype based on your GPU
    device_map="auto",
    max_new_tokens=100,  # Adjust based on desired output length
    num_return_sequences=1,  # Number of samples per generation
    temperature=0.5,  # Controls randomness; lower is more deterministic
    top_p=0.9,  # Nucleus sampling
    repetition_penalty=1.2  # Penalizes repetition
)

prompt = """
Crie exemplos para treino de um modelo que detecta discurso de ódio. Forneça os dados no formato 'texto, label', onde 'label' é 1 para discurso de ódio e 0 caso contrário. Utilize termos ofensivos mascarados com números e caracteres especiais quando apropriado.

Exemplos:
curso de especialização em gestão financeira angola não deixe esta oportunidade passari consultenos em,0
refugiados que a imprensa não mostra,0
está pirralha deveria se preocupar com a fome na áfrica e mundo,1
agora tu ataca néquando queria se promover pela fé vc é uma nojenta,1

# Novos exemplos:
olha esse rapaz parece um m@c@c0, 1
tinha que ser g0rd4 mesmo, 1
pr3t0 só faz m3r.d4, 1
"""

NUM_SAMPLES = 200  # Adjust as needed


def parse_generated_text(generated_text):
    """
    Parses the generated text to extract 'text' and 'label'.
    Assumes the format: 'text,label'
    """
    lines = generated_text.strip().split('\n')
    data = []
    for line in lines:
        # Use regex to split on the last comma to handle commas within text
        match = re.match(r"^(.*),\s*(0|1)$", line.strip())
        if match:
            text, label = match.groups()
            data.append({"text": text.strip(), "label": int(label)})
    return data


# Initialize a list to store generated data
generated_data = []

print("Starting data generation...")

for i in range(NUM_SAMPLES):
    print(f"Generating sample {i + 1}/{NUM_SAMPLES}")

    # Generate text
    generated = pipe(prompt, clean_up_tokenization_spaces=True)

    # Extract the generated text
    generated_text = generated[0]['generated_text']

    # Remove the prompt from the generated text
    new_example = generated_text.replace(prompt, '', 1).strip()

    # Parse the generated example
    parsed = parse_generated_text(new_example)

    if parsed:
        generated_data.extend(parsed)
    else:
        print(f"Failed to parse sample {i + 1}: {new_example}")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(generated_data)

# Drop any duplicate entries
df.drop_duplicates(inplace=True)

# Save the DataFrame to a CSV file
df.to_csv("generated_hate_speech_dataset.csv", index=False, encoding='utf-8')

print("Data generation completed. Dataset saved as 'generated_hate_speech_dataset.csv'.")
