import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
import ipywidgets as widgets
from IPython.display import display

# Descargar y preparar datos (Ejemplo: Fragmentos de una obra de Shakespeare)
text = """
Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance or nature's changing course untrimmed;
But thy eternal summer shall not fade
Nor lose possession of that fair thou ow'st;
Nor shall Death brag thou wander'st in his shade,
When in eternal lines to time thou grow'st:
So long as men can breathe or eyes can see,
So long lives this, and this gives life to thee.
"""

# Preprocesamiento del texto
tokenizer = Tokenizer(char_level=True)  # Tokenización a nivel de caracteres
tokenizer.fit_on_texts([text])
total_chars = len(tokenizer.word_index) + 1  # Total de caracteres únicos
sequences = tokenizer.texts_to_sequences([text])[0]

# Crear secuencias de entrenamiento
sequence_length = 40  # Número de caracteres por secuencia
X = []
y = []
for i in range(0, len(sequences) - sequence_length):
    X.append(sequences[i: i + sequence_length])
    y.append(sequences[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Definir el modelo LSTM
model = Sequential()
model.add(Embedding(total_chars, 50, input_length=sequence_length))  # Toma una secuencia de caracteres (40 en este caso) y la convierte en una secuencia de vectores de 50 dimensiones.
model.add(LSTM(256, return_sequences=False))  # Procesa la secuencia de vectores y genera una salida de 128 dimensiones, capturando las relaciones secuenciales entre los caracteres.
model.add(Dense(total_chars, activation='softmax'))  # Genera una predicción del próximo carácter basado en la salida de la LSTM, con una probabilidad asignada a cada carácter en el vocabulario.

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#ADAM: Ajusta los pesos del modelo de manera eficiente y adaptativa
#Loss: Mide la diferencia entre las predicciones de la red y las etiquetas enteras correctas, usando entropía cruzada categórica para clasificación multiclase.
#Metrica: Mide la precisión de las predicciones del modelo, lo que permite monitorear su rendimiento durante el entrenamiento.

# Entrenar el modelo y guardar el historial
history = model.fit(X, y, epochs=100, batch_size=128)

# Visualizar el modelo
plot_model(model, to_file='rnn_model.png', show_shapes=True, show_layer_names=True)

# Graficar la pérdida (loss) durante el entrenamiento
plt.plot(history.history['loss'])
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()

# Graficar precisión durante el entrenamiento
plt.plot(history.history['accuracy'])
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.show()

# Crear un modelo para visualizar las activaciones de la capa LSTM
input_layer = Input(shape=(sequence_length,))
embedding_layer = model.layers[0](input_layer)  # Capa de embedding
lstm_layer = model.layers[1](embedding_layer)  # Capa LSTM
activation_model = Model(inputs=input_layer, outputs=lstm_layer)

# Función para generar texto y visualizar el proceso
def generar_texto(model, tokenizer, seed_text, max_length, num_chars):
    generated_text = seed_text
    print(f"Texto inicial: '{seed_text}'")
    activations = []  # Para guardar las activaciones
    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_char_index = np.argmax(predicted_probs)
        output_char = tokenizer.index_word[predicted_char_index]

        generated_text += output_char
        
        # Obtener las activaciones de la capa LSTM
        activation = activation_model.predict(token_list)
        activations.append(activation.flatten())  # Guardar las activaciones en una lista

        # Mostrar el estado en cada paso
        print(f"Paso {_+1}: '{generated_text}' (predicho: '{output_char}')")
    
    return generated_text, activations

# Probar generando texto
seed_text = "Shall I compare thee to a "
generated_text, activations = generar_texto(model, tokenizer, seed_text, sequence_length, num_chars=100)

# Configurar el selector interactivo
neuron_selector = widgets.IntSlider(value=0, min=0, max=127, step=1, description='Neurona:', continuous_update=False)
output = widgets.Output()

def plot_neuron(change):
    with output:
        output.clear_output()
        plt.figure(figsize=(10, 6))
        plt.plot([activation[change['new']] for activation in activations], label=f'Neurona {change["new"] + 1}')
        plt.title(f'Activaciones de la neurona {change["new"] + 1}')
        plt.xlabel('Paso de tiempo')
        plt.ylabel('Activación')
        plt.legend()
        plt.show()

neuron_selector.observe(plot_neuron, names='value')

# Mostrar el selector y el gráfico
display(neuron_selector, output)

print("Texto generado:", generated_text)
