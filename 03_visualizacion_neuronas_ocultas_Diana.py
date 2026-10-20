# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.8", "numpy>=1.26", "scikit-learn>=1.4"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    return make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ¿Qué aprende cada neurona oculta?

    En el notebook anterior observamos la frontera producida por toda la red. Sin
    embargo, la capa oculta contiene varias neuronas y cada una aprende parámetros
    diferentes:

    \[
    z_j^{[1]}=w_{j1}^{[1]}x_1+w_{j2}^{[1]}x_2+b_j^{[1]}
    \]

    donde \(z_j^{[1]}=0\) describe una recta. La activación indica en qué lado de esta recta se encuentra cada observación.
    """)
    return


@app.cell
def _(mo):
    neuronas = mo.ui.slider(2, 8, value=4, step=1, label="Número de neuronas")
    ruido = mo.ui.slider(0.02, 0.30, value=0.14, step=0.01, label="Ruido")
    iteraciones = mo.ui.slider(500, 8000, value=4000, step=250, label="Iteraciones")
    learning_rate = mo.ui.slider(0.05, 1.5, value=0.8, step=0.05, label="Learning rate")

    mo.vstack([
        mo.md("### Controles del modelo"),
        mo.hstack([neuronas, ruido]),
        mo.hstack([iteraciones, learning_rate]),
    ])
    return iteraciones, learning_rate, neuronas, ruido


@app.cell
def _(make_moons, ruido):
    # Se generan datos sintéticos
    muestras, etiquetas = make_moons(
        n_samples=250, noise=ruido.value, random_state=12
    )
    X = muestras.T
    Y = etiquetas.reshape(1, -1)
    return X, Y


@app.cell
def _(np):
    def sigmoid(Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def initialize_parameters(n_x, n_h, semilla=5):
        generador = np.random.default_rng(semilla)

        W1 = generador.normal(0, np.sqrt(1 / n_x), (n_h, n_x))
        b1 = np.zeros((n_h, 1))
        W2 = generador.normal(0, np.sqrt(1 / n_h), (1, n_h))
        b2 = np.zeros((1, 1))

        parameters = {
            "W1":W1,
            "b1":b1,
            "W2":W2,
            "b2":b2
        }

        return parameters

    def forward_propagation(X, parameters):

        # parametros, pesos y sesgos
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.matmul(W1,X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.matmul(W2,A1) + b2
        A2 = sigmoid(Z2) # salida

        cache = {
            "Z1":Z1,
            "A1":A1,
            "Z2":Z2,
            "A2":A2
        }

        return A2, cache

    def nn_model(X, Y, n_h, num_iterations, learning_rate):
        """Entrena la red de dos capas utilizando descenso del gradiente."""
        m = X.shape[1]
        parameters = initialize_parameters(X.shape[0], n_h)
        costs = []

        for i in range(num_iterations):
            A2, cache = forward_propagation(X, parameters)
            A2_segura = np.clip(A2, 1e-10, 1 - 1e-10)
            cost = -np.mean(Y * np.log(A2_segura) + (1 - Y) * np.log(1 - A2_segura))

            dZ2 = A2 - Y
            dW2 = np.matmul(dZ2,cache["A1"].T) / m
            db2 = np.sum(dZ2, axis=1, keepdims=True) / m
            dA1 = np.matmul(parameters["W2"].T,dZ2)
            dZ1 = dA1 * (1 - cache["A1"] ** 2)
            dW1 = np.matmul(dZ1, X.T) / m
            db1 = np.sum(dZ1, axis=1, keepdims=True) / m

            parameters["W1"] -= learning_rate * dW1
            parameters["b1"] -= learning_rate * db1
            parameters["W2"] -= learning_rate * dW2
            parameters["b2"] -= learning_rate * db2

            if i % 20 == 0:
                costs.append((i, cost))

        return parameters, costs

    return forward_propagation, nn_model


@app.cell
def _(
    X,
    Y,
    forward_propagation,
    iteraciones,
    learning_rate,
    neuronas,
    nn_model,
    np,
):
    parameters, costs = nn_model(
        X, Y,
        n_h=neuronas.value,
        num_iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    A2, cache = forward_propagation(X, parameters)
    accuracy = float(np.mean((A2 > 0.5) == Y))
    return accuracy, cache, parameters


@app.cell
def _(mo, neuronas):
    neurona = mo.ui.slider(
        1, neuronas.value, value=1, step=1,
        label="Neurona oculta que queremos inspeccionar",
    )
    neurona
    return (neurona,)


@app.cell(hide_code=True)
def _(mo, neurona, parameters):
    indice = neurona.value - 1
    pesos_neurona = parameters["W1"][indice]
    bias_neurona = parameters["b1"][indice, 0]
    peso_salida = parameters["W2"][0, indice]
    mo.md(
        rf"""
        ## Neurona oculta {neurona.value}

        La combinación lineal aprendida por esta neurona es:

        \[
        z_{{{neurona.value}}}^{{[1]}}
        =({pesos_neurona[0]:.3f})x_1
        +({pesos_neurona[1]:.3f})x_2
        +({bias_neurona:.3f})
        \]

        Su peso en la neurona de salida es
        \(w_{{1,{neurona.value}}}^{{[2]}}={peso_salida:.3f}\).
        El signo y la magnitud de este valor indican cómo utiliza la salida final la
        característica construida por esta neurona.

        La combinación final utiliza las activaciones de **todas** las neuronas:

        \[
        z^{{[2]}}=\sum_j w_{{1,j}}^{{[2]}}\tanh(z_j^{{[1]}})+b^{{[2]}},
        \qquad \hat y=\sigma(z^{{[2]}}).
        \]

        La frontera negra corresponde a \(\hat y=0.5\), equivalente a
        \(z^{{[2]}}=0\). El selector cambia la neurona inspeccionada;
        la combinación final siempre utiliza toda la capa oculta.
        """
    )
    return (indice,)


@app.cell
def _(X, Y, forward_propagation, indice, np, parameters, plt):
    eje_x1 = np.linspace(X[0].min() - 0.6, X[0].max() + 0.6, 220)
    eje_x2 = np.linspace(X[1].min() - 0.6, X[1].max() + 0.6, 220)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    X_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    A2_malla, cache_malla = forward_propagation(X_malla, parameters)
    activacion_malla = cache_malla["A1"][indice].reshape(malla_x1.shape)
    Z_neurona = cache_malla["Z1"][indice].reshape(malla_x1.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    mapa_activacion = axes[0].contourf(
        malla_x1, malla_x2, activacion_malla,
        levels=np.linspace(-1, 1, 21), cmap="coolwarm",
    )
    axes[0].contour(
        malla_x1, malla_x2, Z_neurona,
        levels=[0], colors="black", linewidths=2,
    )
    axes[0].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=15, edgecolor="white")
    axes[0].set(title="Activación de la neurona", xlabel="$x_1$", ylabel="$x_2$")
    fig.colorbar(mapa_activacion, ax=axes[0], label="$a_j^{[1]}$")

    for indice_linea in range(parameters["W1"].shape[0]):
        Z_linea = cache_malla["Z1"][indice_linea].reshape(malla_x1.shape)
        axes[1].contour(
            malla_x1, malla_x2, Z_linea, levels=[0],
            linewidths=3 if indice_linea == indice else 1.3,
        )
    axes[1].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=20, edgecolor="white")
    axes[1].set(title="Rectas de todas las neuronas", xlabel="$x_1$", ylabel="$x_2$")

    axes[2].contourf(
        malla_x1, malla_x2, A2_malla.reshape(malla_x1.shape),
        levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.65,
    )
    axes[2].contour(
        malla_x1, malla_x2, A2_malla.reshape(malla_x1.shape),
        levels=[0.5], colors="black", linewidths=2,
    )
    axes[2].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", s=20, edgecolor="white")
    axes[2].set(title="Combinación final", xlabel="$x_1$", ylabel="$x_2$")
    fig.tight_layout()
    fig
    return


@app.cell
def _(cache, neurona, plt):
    fig_hist, ax_hist = plt.subplots(figsize=(7.5, 3.5))
    valores_neurona = cache["A1"][neurona.value - 1]
    ax_hist.hist(valores_neurona, bins=25, color="mediumpurple", edgecolor="white")
    ax_hist.set(
        title=f"Distribución de activaciones: neurona {neurona.value}",
        xlabel="$a_j^{[1]}$", ylabel="Número de observaciones",
    )
    ax_hist.grid(axis="y", alpha=0.2)
    fig_hist.tight_layout()
    fig_hist
    return


@app.cell(hide_code=True)
def _(accuracy, mo):
    mo.callout(
        mo.md(
            f"""
            **Exactitud de la red: {accuracy:.1%}**

            Las neuronas de la capa oculta no deberían interpretarse como clasificadores independientes.
            Cada una construye una característica intermedia (o virtual).
            La neurona de salida es la que aprende a combinar todas estas características creadas.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(X, Y, forward_propagation, mo, np, parameters):
    # Pregunta 2: neuronas parecidas y cuánto aporta cada una
    _A1 = forward_propagation(X, parameters)[1]["A1"]
    _corr = np.corrcoef(_A1)
    _n = _A1.shape[0]
    _filas = []
    for _j in range(_n):
        _q = {k: v.copy() for k, v in parameters.items()}
        _q["W2"][0, _j] = 0.0  # se apaga la contribución de la neurona j
        _acc = float(np.mean((forward_propagation(X, _q)[0] > 0.5) == Y))
        _otras = [(_k, _corr[_j, _k]) for _k in range(_n) if _k != _j]
        _k_max, _c_max = max(_otras, key=lambda t: abs(t[1]))
        _filas.append(
            {
                "Neurona": _j + 1,
                "Peso de salida": f"{parameters['W2'][0, _j]:.2f}",
                "Exactitud sin ella": f"{_acc:.1%}",
                "Más parecida a": _k_max + 1,
                "Correlación": f"{_c_max:+.2f}",
            }
        )

    mo.vstack(
        [
            mo.md(
                "### Redundancia entre neuronas\n\n"
                "Se apaga cada neurona (peso de salida = 0) y se mide cuánto cae la exactitud. "
                "Una correlación cercana a +1 o −1 indica que dos neuronas construyen casi la misma característica."
            ),
            mo.ui.table(_filas, selection=None),
        ]
    )
    return


@app.cell
def _(X, Y, forward_propagation, iteraciones, learning_rate, mo, nn_model, np):
    # Pregunta 3: exactitud según el número de neuronas
    _filas = []
    for _nh in range(2, 9):
        _par, _ = nn_model(X, Y, n_h=_nh, num_iterations=iteraciones.value, learning_rate=learning_rate.value)
        _A2 = forward_propagation(X, _par)[0]
        _Ac = np.clip(_A2, 1e-10, 1 - 1e-10)
        _costo = -np.mean(Y * np.log(_Ac) + (1 - Y) * np.log(1 - _Ac))
        _filas.append(
            {
                "Neuronas": _nh,
                "Exactitud": f"{float(np.mean((_A2 > 0.5) == Y)):.1%}",
                "Costo final": f"{_costo:.4f}",
            }
        )

    mo.vstack([mo.md("### Exactitud según el número de neuronas"), mo.ui.table(_filas, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preguntas

    1. Recorre las neuronas con el control y compara sus rectas y activaciones.
    2. Identifica neuronas con representaciones parecidas. ¿Son todas necesarias?
    3. Aumenta el número de neuronas. ¿La frontera siempre mejora?
    4. Observa el peso de salida de cada neurona. ¿Qué significa un valor negativo?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Respuestas

    *(Los valores numéricos corresponden a los parámetros por defecto: 4 neuronas, ruido = 0.14, 4000 iteraciones, learning rate = 0.8. Las tablas se recalculan al mover los controles.)*

    **1. Comparación de las neuronas.**
    Cada neurona dibuja una recta distinta, \(z_j^{[1]}=0\), y su activación \(\tanh(z_j^{[1]})\) va de −1 a +1 a ambos lados de esa recta, con una transición suave cerca de ella. Con 4 neuronas, por ejemplo, la neurona 1 usa pesos \((-5.29,\,-3.02)\) y la neurona 3 usa \((0.44,\,-0.35)\): la primera tiene una transición muy abrupta y la segunda es casi plana en toda la región de datos. Ninguna recta separa por sí sola las dos lunas; cada una solo "corta" una parte del plano. Lo que separa las clases es la combinación de todas.

    **2. Neuronas con representaciones parecidas.**
    Aparecen en la tabla de redundancia. Con 6 neuronas, las neuronas 2 y 4 tienen correlación de activaciones de −1.00 y pesos de salida de signos opuestos (≈ +5.07 y −5.14): son la misma característica con el signo invertido, porque \(\tanh\) es impar, y la capa de salida lo compensa. Lo mismo pasa con las neuronas 3 y 6 (correlación +0.98). Con 8 neuronas hay varios pares así (2 y 8, 3 y 4, 3 y 6).
    En esos casos no todas son necesarias: al apagar una neurona redundante la exactitud casi no cambia (con 4 neuronas, quitar la 3 deja la exactitud en 100 %), mientras que quitar una neurona clave la hace caer (la 1 la baja a 77 %; con 6 neuronas, quitar la 5 la baja a 77 %). Aun así, la red necesita tener varias neuronas durante el entrenamiento, aunque algunas terminen duplicando a otras.

    **3. ¿La frontera siempre mejora al aumentar neuronas?**
    No. Con estos datos: 2 neuronas dan ≈ 85 %, pero con 3 neuronas ya se alcanza 100 % y de 4 a 8 se mantiene en 100 % con un costo prácticamente igual (≈ 0.002–0.003). Pasar de 2 a 3 neuronas es un salto grande, y después los rendimientos son decrecientes: las neuronas adicionales terminan duplicando características, como se vio en la pregunta 2. Con ruido alto (0.30) también se estanca: ≈ 92 % en entrenamiento y ≈ 91.5 % en un conjunto nuevo desde 3 neuronas, así que más neuronas solo agregan costo de cómputo sin mejorar. (Con redes muy grandes y pocos datos sí puede aparecer sobreajuste, aunque no se observó en estos experimentos.)

    **4. ¿Qué significa un peso de salida negativo?**
    La salida usa \(z^{[2]}=\sum_j w_{1,j}^{[2]}\,a_j^{[1]}+b^{[2]}\). Un peso positivo empuja la predicción hacia la clase 1 donde la neurona está activa (\(a_j>0\)) y hacia la clase 0 donde está en el lado opuesto (\(a_j<0\)). Un peso negativo hace lo contrario: la región donde la neurona vale +1 favorece la clase 0. El signo solo indica la dirección del efecto, y la magnitud indica cuánto importa esa neurona. Por eso el signo de una neurona aislada no tiene significado propio: invertir el signo de \((w_j, b_j)\) invierte \(a_j\), y basta cambiar el signo de su peso de salida para obtener exactamente la misma red. Lo que importa es el producto \(w_{1,j}^{[2]}\,a_j^{[1]}\).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
