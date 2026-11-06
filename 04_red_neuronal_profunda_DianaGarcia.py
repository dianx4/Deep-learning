# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.8", "numpy>=1.26", "scikit-learn>=1.4"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_circles, make_moons

    return make_circles, make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Red neuronal profunda

    Hasta el momento, hemos escrito explícitamente los parámetros `W1`, `b1`, `W2` y `b2`. Para poder construir redes con más capas necesitamos generalizar aún más el código.

    Para este propósito, describiremos la arquitectura mediante una lista:

    ```python
    layer_dims = [n_x, n_1, n_2, ..., n_y]
    ```

    Para cada capa \(l\), la propagación hacia adelante utiliza:

    \[
    Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]},\qquad
    A^{[l]}=g^{[l]}(Z^{[l]})
    \]

    donde \(A^{[0]}=X\). La última capa continuará utilizando una sigmoide porque estamos resolviendo un problema de clasificación binaria.
    """)
    return


@app.cell
def _(mo):
    conjunto = mo.ui.dropdown(
        options={"Dos lunas": "lunas", "Círculos concéntricos": "circulos"},
        value="Círculos concéntricos", label="Conjunto de datos",
    )
    capas_ocultas = mo.ui.slider(1, 4, value=2, step=1, label="Capas ocultas")
    ancho = mo.ui.slider(2, 16, value=6, step=1, label="Neuronas por capa")
    activacion = mo.ui.dropdown(
        options={"Tangente hiperbólica": "tanh", "ReLU": "relu"},
        value="Tangente hiperbólica", label="Activación",
    )
    learning_rate = mo.ui.slider(0.01, 1.2, value=0.4, step=0.01, label="Learning rate")
    iteraciones = mo.ui.slider(500, 10000, value=4000, step=250, label="Iteraciones")

    mo.vstack([
        mo.md("### Arquitectura y entrenamiento"),
        conjunto,
        mo.hstack([capas_ocultas, ancho, activacion]),
        mo.hstack([learning_rate, iteraciones]),
    ])
    return (
        activacion,
        ancho,
        capas_ocultas,
        conjunto,
        iteraciones,
        learning_rate,
    )


@app.cell
def _(conjunto, make_circles, make_moons):
    if conjunto.value == "circulos":
        muestras, etiquetas = make_circles(
            n_samples=300, noise=0.08, factor=0.42, random_state=18
        )
    else:
        muestras, etiquetas = make_moons(
            n_samples=300, noise=0.18, random_state=18
        )
    X = muestras.T
    Y = etiquetas.reshape(1, -1)
    return X, Y


@app.cell
def _(ancho, capas_ocultas):
    layer_dims = [2] + [ancho.value] * capas_ocultas.value + [1]
    return (layer_dims,)


@app.cell(hide_code=True)
def _(layer_dims, mo):
    arquitectura = " → ".join(str(valor) for valor in layer_dims)
    filas_dimensiones = []
    for capa in range(1, len(layer_dims)):
        filas_dimensiones.append(
            f"| {capa} | `W{capa}` | "
            f"$({layer_dims[capa]}, {layer_dims[capa - 1]})$ | "
            f"`b{capa}` | $({layer_dims[capa]},1)$ |"
        )
    tabla_dimensiones = "\n".join(filas_dimensiones)
    mo.md(
        rf"""
        ## Arquitectura seleccionada

        \[
        {arquitectura}
        \]

        | Capa | Pesos | Dimensión | Bias | Dimensión |
        |---:|---|---:|---|---:|
        {tabla_dimensiones}

        Las dimensiones se deducen automáticamente de `layer_dims`; ya no es
        necesario escribir una función diferente para cada profundidad.
        """
    )
    return


@app.cell
def _(np):
    def sigmoid(Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def hidden_activation(Z, nombre):

        if nombre == "tanh":
            return np.tanh(Z)

        else:
            return np.maximum(0,Z)

    def hidden_derivative(Z, A, nombre):

        if nombre == "tanh":
            return 1 - A**2
        else:
            return (Z > 0).astype(float)

    def initialize_parameters_deep(layer_dims, nombre="tanh", semilla=21):
        """
        Inicializa los parámetros para una red de L capas.

        Argumentos:
        layer_dims -- lista con el número de unidades de cada capa
        nombre -- activación utilizada en las capas ocultas

        Retorna:
        parameters -- diccionario que contiene W1, b1, ..., WL, bL
        """
        generador = np.random.default_rng(semilla)
        parameters = {}
        L = len(layer_dims) - 1

        for l in range(1, L + 1):
            factor = 2 if nombre == "relu" and l < L else 1
            escala = np.sqrt(factor / layer_dims[l - 1])
            parameters[f"W{l}"] = generador.normal(
                0, escala, (layer_dims[l], layer_dims[l - 1])
            )
            parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        return parameters

    def forward_propagation_deep(X, parameters, nombre="tanh"):
        """
        Implementa la propagación hacia adelante para una red de L capas.

        El cache conserva los valores necesarios para backpropagation.
        """
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_previa = A
            Z = parameters[f"W{l}"] @ A_previa + parameters[f"b{l}"]
            A = hidden_activation(Z, nombre)
            caches.append({"A_prev": A_previa, "Z": Z, "A": A})

        A_previa = A
        ZL = parameters[f"W{L}"] @ A_previa + parameters[f"b{L}"]
        AL = sigmoid(ZL)
        caches.append({"A_prev": A_previa, "Z": ZL, "A": AL})
        return AL, caches

    def compute_cost_deep(AL, Y):
        AL_segura = np.clip(AL, 1e-10, 1 - 1e-10)
        return -np.mean(Y * np.log(AL_segura) + (1 - Y) * np.log(1 - AL_segura))

    def backward_propagation_deep(AL, Y, parameters, caches, nombre="tanh"):
        """Implementa backpropagation desde la salida hasta la primera capa."""
        grads = {}
        m = Y.shape[1]
        L = len(caches)

        dZ = AL - Y
        grads[f"dW{L}"] = (dZ @ caches[L - 1]["A_prev"].T) / m
        grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True) / m
        dA_previa = parameters[f"W{L}"].T @ dZ

        for l in reversed(range(1, L)):
            cache = caches[l - 1]
            dZ = dA_previa * hidden_derivative(cache["Z"], cache["A"], nombre)
            grads[f"dW{l}"] = (dZ @ cache["A_prev"].T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            dA_previa = parameters[f"W{l}"].T @ dZ

        return grads

    def update_parameters_deep(parameters, grads, learning_rate):
        L = len(parameters) // 2
        for l in range(1, L + 1):
            parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
        return parameters

    def nn_model_deep(X, Y, layer_dims, nombre, num_iterations, learning_rate):
        """Integra el loop de entrenamiento de una red profunda."""
        parameters = initialize_parameters_deep(layer_dims, nombre)
        costs = []

        for i in range(num_iterations):
            AL, caches = forward_propagation_deep(X, parameters, nombre)
            cost = compute_cost_deep(AL, Y)
            grads = backward_propagation_deep(AL, Y, parameters, caches, nombre)
            parameters = update_parameters_deep(parameters, grads, learning_rate)
            if i % 20 == 0:
                costs.append((i, cost))

        return parameters, costs

    return forward_propagation_deep, nn_model_deep


@app.cell
def _(
    X,
    Y,
    activacion,
    forward_propagation_deep,
    iteraciones,
    layer_dims,
    learning_rate,
    nn_model_deep,
    np,
):
    parameters, costs = nn_model_deep(
        X, Y,
        layer_dims=layer_dims,
        nombre=activacion.value,
        num_iterations=iteraciones.value,
        learning_rate=learning_rate.value,
    )
    AL, caches = forward_propagation_deep(X, parameters, activacion.value)
    accuracy = float(np.mean((AL > 0.5) == Y))
    return accuracy, caches, costs, parameters


@app.cell
def _(X, Y, activacion, costs, forward_propagation_deep, np, parameters, plt):
    eje_x1 = np.linspace(X[0].min() - 0.5, X[0].max() + 0.5, 240)
    eje_x2 = np.linspace(X[1].min() - 0.5, X[1].max() + 0.5, 240)
    malla_x1, malla_x2 = np.meshgrid(eje_x1, eje_x2)
    X_malla = np.vstack((malla_x1.ravel(), malla_x2.ravel()))
    AL_malla, _caches_malla = forward_propagation_deep(
        X_malla, parameters, activacion.value
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].contourf(
        malla_x1, malla_x2, AL_malla.reshape(malla_x1.shape),
        levels=np.linspace(0, 1, 21), cmap="RdBu_r", alpha=0.7,
    )
    axes[0].contour(
        malla_x1, malla_x2, AL_malla.reshape(malla_x1.shape),
        levels=[0.5], colors="black", linewidths=2,
    )
    axes[0].scatter(X[0], X[1], c=Y.ravel(), cmap="bwr", edgecolor="white", s=28)
    axes[0].set(title="Frontera de decisión", xlabel="$x_1$", ylabel="$x_2$")

    pasos, valores = zip(*costs)
    axes[1].plot(pasos, valores, color="purple")
    axes[1].set(title="Loop de entrenamiento", xlabel="Iteración", ylabel="Costo")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(accuracy, caches, layer_dims, mo):
    formas_activaciones = [cache["A"].shape for cache in caches]
    mo.callout(
        mo.md(
            f"""
            **Exactitud: {accuracy:.1%}**

            Arquitectura utilizada: `{layer_dims}`

            Formas de las activaciones: `{formas_activaciones}`

            Aun si el número de capas cambia, el loop exterior conserva las mismas cuatro etapas:
            forward propagation, costo (error), backpropagation y actualización de parámetros w,b.
            """
        ),
        kind="success" if accuracy >= 0.85 else "warn",
    )
    return


@app.cell
def _(
    X,
    Y,
    ancho,
    forward_propagation_deep,
    iteraciones,
    learning_rate,
    mo,
    nn_model_deep,
    np,
):
    # Preguntas 1 y 2: profundidad y activación con ancho constante
    _filas = []
    for _h in range(1, 5):
        _dims = [2] + [ancho.value] * _h + [1]
        _fila = {"Capas ocultas": _h}
        for _nombre, _etiqueta in [("tanh", "tanh"), ("relu", "ReLU")]:
            _par, _ = nn_model_deep(
                X, Y, layer_dims=_dims, nombre=_nombre,
                num_iterations=iteraciones.value, learning_rate=learning_rate.value,
            )
            _AL = forward_propagation_deep(X, _par, _nombre)[0]
            _AL_s = np.clip(_AL, 1e-10, 1 - 1e-10)
            _costo = -np.mean(Y * np.log(_AL_s) + (1 - Y) * np.log(1 - _AL_s))
            _fila[f"Exactitud {_etiqueta}"] = f"{float(np.mean((_AL > 0.5) == Y)):.1%}"
            _fila[f"Costo {_etiqueta}"] = f"{_costo:.4f}"
        _filas.append(_fila)

    mo.vstack(
        [
            mo.md(
                f"### Profundidad y activación (ancho constante = {ancho.value} neuronas por capa)\n\n"
                "Entrena 8 redes con los controles actuales; puede tardar unos segundos."
            ),
            mo.ui.table(_filas, selection=None),
        ]
    )
    return


@app.cell
def _(layer_dims, mo):
    # Pregunta 4: número de parámetros de la arquitectura seleccionada
    _filas = []
    _total = 0
    for _l in range(1, len(layer_dims)):
        _pesos = layer_dims[_l] * layer_dims[_l - 1]
        _bias = layer_dims[_l]
        _total += _pesos + _bias
        _filas.append(
            {
                "Capa": _l,
                "Pesos W": f"{layer_dims[_l]} × {layer_dims[_l - 1]} = {_pesos}",
                "Bias b": _bias,
                "Subtotal": _pesos + _bias,
            }
        )

    mo.vstack(
        [
            mo.md(f"### Parámetros de la arquitectura `{layer_dims}`"),
            mo.ui.table(_filas, selection=None),
            mo.md(f"**Total de parámetros entrenables: {_total}**"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## De una capa oculta a una red profunda

    Esta generalización consistió en sustituir el código específico para cada capa por ciclos que recorren la variable `layer_dims`. Durante *forward propagation* se guardó un `cache` por capa. Durante el proceso de *backpropagation* recorremos estos valores en orden inverso aplicando la regla de la cadena.

    Hay que observar que más profundidad en una red neuronal no garantiza automáticamente un mejor resultado. En el proceso participan varios factores:

    - la función de activación
    - la inicialización de los parámetros
    - la tasa de aprendizaje (*learning rate*)
    - el número de iteraciones
    - la cantidad y calidad de los datos

    ### Preguntas

    1. Mantén un número de neuronas constante y aumenta la profundidad.
    2. Compara `tanh` y `ReLU` con la misma arquitectura.
    3. Busca una configuración que no converja y explica la causa probable.
    4. Calcula el número total de parámetros de la arquitectura seleccionada.
    5. Relaciona cada función de este notebook con `Linear`, la activación, la
       función de pérdida y `optimizer.step()` en PyTorch (opcional).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Respuestas



    **1. Ancho constante, más profundidad.**

    | Capas ocultas | Círculos, tanh (costo) | Círculos, ReLU (costo) | Lunas, tanh (exactitud) | Lunas, ReLU (exactitud) |
    |---:|---|---|---|---|
    | 1 | 100 % (0.0125) | 100 % (0.0026) | 98.7 % | 87.0 % |
    | 2 | 100 % (0.0006) | 100 % (0.0002) | 99.0 % | 99.3 % |
    | 3 | 100 % (0.0002) | 100 % (0.0001) | 99.7 % | 99.3 % |
    | 4 | 100 % (0.0001) | 100 % (≈ 0.0000) | 99.7 % | 99.7 % |

    En círculos la exactitud ya es 100 % con una capa, y más profundidad solo reduce el costo. En lunas la exactitud mejora poco al pasar de 2 a 4 capas. La mejora no es estrictamente monótona: con lunas y ReLU el costo con 4 capas (≈ 0.042) es mayor que con 3 (≈ 0.009). En conclusión, más capas dan más capacidad, pero con estos datos pequeños y sencillos los rendimientos son decrecientes, como se menciona al final del notebook.

    **2. `tanh` frente a `ReLU`.**
    Con círculos ambas llegan a 100 % y ReLU alcanza costos algo menores. Con lunas y una sola capa oculta ReLU es peor (87 % frente a 98.7 %): con 6 unidades solo puede armar una frontera con pocos tramos rectos, mientras que `tanh` produce curvas suaves. Desde 2 capas ya son equivalentes. La diferencia visible está en la forma de la frontera: con ReLU es *lineal por tramos* (con aristas) y con `tanh` es suave. ReLU también exige cuidado con las neuronas "muertas" (ver pregunta 3).

    **3. Configuración que no converge.**
    Con círculos, 4 capas ocultas de solo 2 neuronas y learning rate 0.4:
    - **ReLU:** exactitud 50 % y costo 0.6931 (= ln 2) sin moverse. Desde la segunda capa oculta todas las activaciones son 0 (*ReLU muerta*): como la derivada de ReLU es 0 cuando \(Z\le 0\), el gradiente es exactamente 0, no se actualiza nada y la salida es 0.5 para todos los puntos. Con lunas y el mismo ancho pasa lo mismo (50 %).
    - **tanh:** el costo se queda en 0.6932 (exactitud 27.7 %, peor que el azar porque la salida es ≈ 0.5 para todos y el signo de las diferencias es arbitrario). Aquí el gradiente es muy pequeño (≈ 1e-5 al final) y más pequeño aún en las primeras capas (*gradiente que se desvanece*), y además el cuello de botella de 2 neuronas hace difícil propagar información. La red queda en una meseta.

    Otras causas que se pueden provocar: un learning rate muy pequeño (con 0.01 y `tanh` el costo sigue en 0.22 tras 4000 iteraciones: no ha convergido aunque la exactitud sea 99 %) o muy pocas neuronas (1 capa de 2 neuronas llega solo a 80–85 %). En estos datos, un learning rate alto (1.2) no hizo divergir la red, así que esa causa no se pudo reproducir aquí.

    **4. Número de parámetros.**
    Para cada capa \(l\) hay \(n_l \cdot n_{l-1}\) pesos y \(n_l\) sesgos, de modo que

    \[
    \text{total}=\sum_{l=1}^{L} n_l\,(n_{l-1}+1).
    \]

    Para la arquitectura por defecto `[2, 6, 6, 1]`: \(6\cdot 2+6=18\), \(6\cdot 6+6=42\) y \(1\cdot 6+1=7\), es decir, 67 parámetros. Con 1, 2, 3 y 4 capas ocultas de 6 neuronas son 25, 67, 109 y 151 parámetros respectivamente. La celda de parámetros calcula el total para la arquitectura seleccionada.

    **5. Relación con PyTorch (opcional).**

    | Este notebook | PyTorch |
    |---|---|
    | `initialize_parameters_deep` (crea `W`, `b`; escala \(\sqrt{2/n}\) para ReLU) | `nn.Linear(n_in, n_out)` crea `weight` y `bias`; la inicialización se controla con `nn.init` (p. ej. `kaiming_normal_` para ReLU) |
    | `hidden_activation` | `nn.Tanh()` o `nn.ReLU()` |
    | `sigmoid` en la última capa | `nn.Sigmoid()` (o se omite y se usa `BCEWithLogitsLoss`) |
    | `forward_propagation_deep` | `model(X)` (método `forward`); los `caches` los guarda *autograd* internamente |
    | `compute_cost_deep` | `nn.BCELoss()` (o `nn.BCEWithLogitsLoss()`, más estable que el `clip`) |
    | `backward_propagation_deep` y `hidden_derivative` | `loss.backward()` (diferenciación automática) |
    | `update_parameters_deep` | `optimizer.step()` con `torch.optim.SGD(model.parameters(), lr=...)` |
    | `nn_model_deep` (el ciclo) | el *training loop*: `zero_grad()`, forward, pérdida, `backward()`, `step()` |

    ```python
    import torch.nn as nn, torch

    modelo = nn.Sequential(
        nn.Linear(2, 6), nn.Tanh(),
        nn.Linear(6, 6), nn.Tanh(),
        nn.Linear(6, 1),                 # la sigmoide va dentro de la pérdida
    )
    perdida = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=0.4)

    for i in range(4000):
        optimizador.zero_grad()
        salida = modelo(X)               # X con forma (m, 2) en PyTorch
        costo = perdida(salida, Y)       # Y con forma (m, 1)
        costo.backward()
        optimizador.step()
    ```

    Una diferencia práctica: PyTorch pone las observaciones en filas (\(m\times n_x\)), mientras que este curso las pone en columnas (\(n_x\times m\)); por eso `weight` tiene forma `(n_out, n_in)`, igual que `W[l]`, pero se calcula \(XW^\top+b\).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
