# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "matplotlib>=3.8",
#     "numpy>=1.26",
#     "scikit-learn>=1.4",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from sklearn.datasets import make_moons

    return ListedColormap, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Límites de una sola neurona

    En la parte anterior habíamos construido un perceptrón con una función de activación (sigmoide) para resolver un problema de clasificación binaria. Recordamos que la salida del perceptrón se calculaba como,

    \begin{equation}
    A = \sigma (WX + b)
    \end{equation}

    donde $\sigma (z) = \frac{1}{1+e^{-z}}$ es la función sigmoide.

    Normalmente se toma como umbral de decisión cuando $A > 0.5$, es decir, cuando la $WX + b = 0$. Por lo tanto, **una sola neurona únicamente puede construir una frontera de decisión lineal**
    """)
    return


@app.cell
def _(mo):
    # Definimos algunos widgets con marimo

    conjunto = mo.ui.dropdown(
        options={
            "Dos grupos separables": "lineal",
            "Compuerta XOR": "xor",
            "Dos lunas": "lunas",
        },
        value="Compuerta XOR",
        label="Conjunto de datos",
    )
    ruido = mo.ui.slider(
        start=0.0,
        stop=0.35,
        step=0.01,
        value=0.08,
        label="Ruido",
    )
    iteraciones = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1500,
        label="Iteraciones",
    )
    learning_rate = mo.ui.slider(
        start=0.01,
        stop=2.0,
        step=0.01,
        value=0.5,
        label="Learning rate",
    )

    mo.vstack(
        [
            mo.md("### Controles del experimento"),
            conjunto,
            ruido,
            iteraciones,
            learning_rate,
        ]
    )
    return conjunto, iteraciones, learning_rate, ruido


@app.cell
def _(conjunto, make_moons, np, ruido):
    def construir_datos(nombre, ruido, semilla=42):
        """
        Construye un problema de clasificación binaria con observaciones en columnas.
        """
        generador = np.random.default_rng(semilla)

        if nombre == "lineal":
            clase_0 = generador.normal((-1.2, -1.0), ruido + 0.25, (80, 2))
            clase_1 = generador.normal((1.2, 1.0), ruido + 0.25, (80, 2))
            muestras = np.vstack((clase_0, clase_1))
            etiquetas = np.hstack((np.zeros(80), np.ones(80)))
        elif nombre == "xor":
            centros = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            muestras = np.repeat(centros, 40, axis=0)
            muestras = muestras + generador.normal(0, ruido, muestras.shape)
            etiquetas = np.repeat([0, 1, 1, 0], 40)
        else:
            muestras, etiquetas = make_moons(
                n_samples=200,
                noise=ruido,
                random_state=semilla,
            )

        return muestras.T, etiquetas.reshape(1, -1)

    X, y = construir_datos(conjunto.value, ruido.value)
    return X, construir_datos, y


@app.cell(hide_code=True)
def _(X, mo, y):
    mo.md(rf"""
    ## Forma de los datos

    Organizamos los datos utilizando la misma convención que en los notebooks anteriores

    \[
    X\in\mathbb{{R}}^{{n_x\times m}},\qquad
    Y\in\mathbb{{R}}^{{1\times m}}
    \]

    Para el conjunto seleccionado:

    - `X.shape = {X.shape}`
    - `y.shape = {y.shape}`
    - tenemos \(m={X.shape[1]}\) observaciones de entrenamiento.
    """)
    return


@app.cell
def _(np):
    def sigmoid(Z):
        """
        Aplica la función sigmoide elemento por elemento.
        """
        Z_seguro = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Z_seguro))

    def initialize_parameters(n_x, semilla=42):
        """
        Inicializa los pesos de una única neurona sigmoide.
        """
        generador = np.random.default_rng(semilla)

        W = generador.normal(0, 0.01, (1, n_x))
        b = np.zeros((1, 1))

        parameters = {
            "W":W,
            "b":b
        }

        return parameters

    def forward_propagation(X, parameters):
        """
        Calcula Z=WX+b y posteriormente A=sigmoid(Z).
        """
        W = parameters["W"]
        b = parameters["b"]

        Z = np.matmul(W,X) + b

        y_hat = sigmoid(Z)
    
        return y_hat

    def loss(y_hat, y):
        """
        Calcula la entropía cruzada binaria promedio.
        """

        A_segura = np.clip(y_hat, 1e-10, 1 - 1e-10)

        loss = -np.mean(y * np.log(A_segura) + (1-y) * np.log(1-A_segura))

        return loss

    def backward_propagation(y_hat, X, y):
        """
        Calcula los gradientes de W y b.
        """
        m = X.shape[1]
        dZ = y_hat - y

        dW = np.matmul(dZ,X.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        grads = {
            "dW": dW,
            "db":db
        }
        return grads

    def update_parameters(parameters, grads, learning_rate=0.5):
        """
        Actualiza los parámetros utilizando la regla de descenso del gradiente
        """

        # pesos w y sesgo b
        W = parameters["W"]
        b = parameters["b"]

        # gradiente
        dW = grads["dW"]
        db = grads["db"]

        # descenso del gradiente
        W = W - learning_rate*dW
        b = b - learning_rate*db

        parameters = {
            "W":W,
            "b":b
        }

        return parameters

    def nn_model(X, y, iterations=1000, learning_rate=0.5):
        """
        Integra forward, costo, backward y actualización de parámetros.
        """
        parameters = initialize_parameters(X.shape[0])
        costs = []

        # ciclo de entrenamiento
        for i in range(iterations):
            y_hat = forward_propagation(X=X, parameters=parameters)
            error = loss(y_hat=y_hat, y=y)
            grads = backward_propagation(y_hat=y_hat, X=X, y=y)
            parameters = update_parameters(parameters=parameters, grads=grads, learning_rate=learning_rate)

            if i % 10 == 0:
                costs.append((i, error))

        return parameters, costs

    return forward_propagation, nn_model, sigmoid


@app.cell
def _(X, forward_propagation, iteraciones, learning_rate, nn_model, np, y):
    parameters, costs = nn_model(
        X,
        y,
        iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    probabilities = forward_propagation(X, parameters)
    predictions = (probabilities > 0.5).astype(int)
    accuracy = float(np.mean(predictions == y))
    return accuracy, costs, parameters


@app.cell
def _(ListedColormap, X, costs, np, parameters, plt, y):
    x1_min, x1_max = X[0].min() - 0.5, X[0].max() + 0.5
    x2_min, x2_max = X[1].min() - 0.5, X[1].max() + 0.5
    eje_x1 = np.linspace(x1_min, x1_max, 250)
    eje_x2 = np.linspace(x2_min, x2_max, 250)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    puntos_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    Z_malla = parameters["W"] @ puntos_malla + parameters["b"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].contourf(
        malla_x1,
        malla_x2,
        (Z_malla.reshape(malla_x1.shape) > 0).astype(int),
        alpha=0.22,
        cmap=ListedColormap(["royalblue", "tomato"]),
    )
    axes[0].contour(
        malla_x1,
        malla_x2,
        Z_malla.reshape(malla_x1.shape),
        levels=[0],
        colors="black",
        linewidths=2,
    )
    axes[0].scatter(
        X[0], X[1], c=y.ravel(),
        cmap=ListedColormap(["blue", "red"]), edgecolor="white",
    )
    axes[0].set(title="Frontera aprendida", xlabel="$x_1$", ylabel="$x_2$")

    pasos, valores_cost = zip(*costs)
    axes[1].plot(pasos, valores_cost, color="purple")
    axes[1].set(title="Loop de entrenamiento", xlabel="Iteración", ylabel="Costo")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()

    fig
    return


@app.cell(hide_code=True)
def _(accuracy, conjunto, mo):
    interpretacion = (
        "En este caso una recta es suficiente para separar las clases."
        if conjunto.value == "lineal"
        else "Aunque el costo disminuya, una sola recta no puede representar la geometría completa del problema."
    )
    mo.callout(
        mo.md(
            rf"""
            **Exactitud obtenida: {accuracy:.1%}**

            {interpretacion}

            La limitación no se debe necesariamente al número de iteraciones. Se
            debe a la estructura del modelo: la frontera \(WX+b=0\) siempre es una
            recta. Esto motiva la incorporación de una **capa oculta con varias
            neuronas**.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(
    construir_datos,
    forward_propagation,
    iteraciones,
    learning_rate,
    mo,
    nn_model,
    np,
    ruido,
):
    # Pregunta 1: comparación de los tres conjuntos con los mismos hiperparámetros
    _filas = []
    for _clave, _nombre in [("lineal", "Dos grupos separables"), ("xor", "Compuerta XOR"), ("lunas", "Dos lunas")]:
        _X, _y = construir_datos(_clave, ruido.value)
        _par, _costos = nn_model(_X, _y, iterations=iteraciones.value, learning_rate=learning_rate.value)
        _pred = (forward_propagation(_X, _par) > 0.5).astype(int)
        _filas.append(
            {
                "Conjunto": _nombre,
                "Exactitud": f"{float(np.mean(_pred == _y)):.1%}",
                "Costo final": f"{_costos[-1][1]:.4f}",
            }
        )

    mo.vstack([mo.md("### Comparación de los tres conjuntos (mismos hiperparámetros)"), mo.ui.table(_filas, selection=None)])
    return


@app.cell
def _(
    construir_datos,
    forward_propagation,
    learning_rate,
    mo,
    nn_model,
    np,
    ruido,
):
    # Pregunta 2: XOR con más iteraciones
    _X, _y = construir_datos("xor", ruido.value)
    _filas = []
    for _it in [100, 1500, 5000, 20000]:
        _par, _costos = nn_model(_X, _y, iterations=_it, learning_rate=learning_rate.value)
        _pred = (forward_propagation(_X, _par) > 0.5).astype(int)
        _filas.append(
            {
                "Iteraciones": _it,
                "Exactitud": f"{float(np.mean(_pred == _y)):.1%}",
                "Costo final": f"{_costos[-1][1]:.4f}",
            }
        )

    mo.vstack([mo.md("### XOR: efecto de las iteraciones"), mo.ui.table(_filas, selection=None)])
    return


@app.cell
def _(
    construir_datos,
    forward_propagation,
    iteraciones,
    mo,
    nn_model,
    np,
    ruido,
):
    # Pregunta 3: XOR y lunas con distintos learning rates
    _filas = []
    for _clave, _nombre in [("xor", "XOR"), ("lunas", "Dos lunas")]:
        _X, _y = construir_datos(_clave, ruido.value)
        for _lr in [0.01, 0.1, 0.5, 2.0]:
            _par, _costos = nn_model(_X, _y, iterations=iteraciones.value, learning_rate=_lr)
            _pred = (forward_propagation(_X, _par) > 0.5).astype(int)
            _filas.append(
                {
                    "Conjunto": _nombre,
                    "Learning rate": _lr,
                    "Exactitud": f"{float(np.mean(_pred == _y)):.1%}",
                    "Costo final": f"{_costos[-1][1]:.4f}",
                }
            )

    mo.vstack([mo.md("### Efecto del learning rate"), mo.ui.table(_filas, selection=None)])
    return


@app.cell
def _(construir_datos, mo, np, ruido, sigmoid):
    # Pregunta 4: agregar variables de entrada a XOR
    _X, _y = construir_datos("xor", ruido.value)
    _g = np.random.default_rng(0)

    def _entrenar(X, y, it=5000, lr=0.5):
        W = np.random.default_rng(42).normal(0, 0.01, (1, X.shape[0]))
        b = np.zeros((1, 1))
        m = X.shape[1]
        for _ in range(it):
            dZ = sigmoid(W @ X + b) - y
            W = W - lr * (dZ @ X.T) / m
            b = b - lr * dZ.sum(axis=1, keepdims=True) / m
        return float(np.mean((sigmoid(W @ X + b) > 0.5) == y))

    _variantes = {
        "Solo $x_1, x_2$": _X,
        "+ 3 variables aleatorias (ruido)": np.vstack((_X, _g.normal(size=(3, _X.shape[1])))),
        "+ combinaciones lineales de $x_1, x_2$": np.vstack((_X, 2 * _X[0:1] + 1, _X[1:2] - _X[0:1])),
        "+ variable no lineal $x_1 x_2$": np.vstack((_X, _X[0:1] * _X[1:2])),
    }
    _filas = [{"Entradas": k, "Exactitud": f"{_entrenar(v, _y):.1%}"} for k, v in _variantes.items()]

    mo.vstack([mo.md("### XOR con variables adicionales"), mo.ui.table(_filas, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preguntas sobre el tema

    1. Selecciona los tres conjuntos y compara la exactitud.
    2. Aumenta el número de iteraciones en XOR. ¿Desaparece la limitación?
    3. Cambia el *learning rate*. ¿Qué parte del resultado corresponde a la
       optimización y qué parte corresponde a la capacidad del modelo?
    4. Explica por qué agregar más variables de entrada no resolvería por sí solo
       el problema XOR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Respuestas

    *(Los números citados corresponden a los valores por defecto: ruido = 0.08, 1500 iteraciones, learning rate = 0.5. Las tablas de arriba se recalculan al mover los controles.)*

    **1. Comparación de los tres conjuntos.**

    | Conjunto | Exactitud | Costo final |
    |---|---|---|
    | Dos grupos separables | 100 % | ≈ 0.001 |
    | Compuerta XOR | 50 % | ≈ 0.693 |
    | Dos lunas | ≈ 86.5 % | ≈ 0.27 |

    En el conjunto lineal una recta separa perfectamente las clases. En XOR la exactitud es la del azar. En las lunas la recta acierta en buena parte de los puntos, pero los que están en la zona donde las lunas se "abrazan" siempre quedan mal clasificados.

    **2. ¿Desaparece la limitación en XOR al aumentar las iteraciones?**
    No. Con 100, 1500, 5000 o incluso 50 000 iteraciones la exactitud se queda en 50 % y el costo en $0.6931 \approx \ln 2$. Ese valor es el costo de un modelo que responde $A = 0.5$ para todos los puntos: por la simetría de XOR, el mejor ajuste lineal es no discriminar nada (los pesos se quedan casi en cero). El descenso del gradiente ya llegó al mínimo posible *dentro de la familia de rectas*; entrenar más no cambia la familia.

    **3. Learning rate: ¿optimización o capacidad?**
    - En XOR el resultado es el mismo para todos los valores probados (0.01 a 2.0): 50 % y costo $\approx 0.693$. Aquí el learning rate es irrelevante porque el límite es de capacidad.
    - En lunas sí cambia algo: con un learning rate muy pequeño (0.01) el modelo aún no convergió en 1500 iteraciones (≈ 82.5 %, costo ≈ 0.36), mientras que con 0.5 o 2.0 llega al mismo mínimo (≈ 86.5 %, costo ≈ 0.27).

    Es decir: el learning rate (y las iteraciones) determinan qué tan rápido y qué tan cerca llegamos al mejor modelo posible de la familia (optimización). La forma de la familia, que aquí es una recta, determina cuál es ese mejor modelo (capacidad). Ningún ajuste de optimización puede superar el techo impuesto por la capacidad.

    **4. ¿Por qué más variables de entrada no resuelven XOR por sí solas?**
    Porque con $n$ entradas la frontera $WX + b = 0$ sigue siendo un hiperplano, solo que en un espacio de mayor dimensión. Si las variables nuevas son ruido o combinaciones lineales de $x_1, x_2$, no aportan información que ayude a separar XOR: en las pruebas la exactitud se queda en ≈ 50 % (54 % con ruido, por sobreajuste casual).

    Matiz importante: si la variable nueva es una transformación no lineal útil, como $x_1 x_2$, entonces XOR se resuelve (100 %), porque en el espacio $(x_1, x_2, x_1x_2)$ las clases sí son separables por un plano. Esa es justamente la idea de la capa oculta: en lugar de que nosotros diseñemos esas variables a mano, las neuronas ocultas aprenden las transformaciones no lineales necesarias.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
