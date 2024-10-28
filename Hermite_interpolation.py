import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import dash_latex as dl
import dash_katex
from dash.dependencies import State, MATCH, ALL
from dash.exceptions import PreventUpdate


# http://127.0.0.1:8050/

def evaluate_polynomial(coeffs, x_values):
    result = np.polyval(coeffs, x_values)
    return result


def evaluate_polynomial_derivative(coeffs, x_values):
    polynomial = np.poly1d(coeffs)
    polynomial_derivative = np.polyder(polynomial)
    result = np.polyval(polynomial_derivative, x_values)
    return result


def interpolation(node):
    n = node.shape[0]

    solution_matrix = np.empty((0, 2 * n + 1))
    for j in range(n):
        temp = np.hstack([np.power(node[j, 0], [i for i in range(2 * n)]), [node[j, 1]]])  # from 0 until m=2n-1
        solution_matrix = np.vstack([solution_matrix, temp])

    for k in range(n):
        temp = np.power(node[k, 0], [i for i in range(2 * n - 1)]) * [i for i in range(1, 2 * n)]  # from 1 to m-1=2n-2
        temp = np.hstack([[0], temp, [node[k, 2]]])
        solution_matrix = np.vstack([solution_matrix, temp])

    B = solution_matrix.T[-1]
    H = solution_matrix.T[0:-1].T

    a = np.linalg.solve(H, B)

    return a[::-1]


def polynomial_to_latex(coeffs, precision=2):
    terms = []
    coeffs = coeffs[::-1]
    for i, coeff in enumerate(coeffs):
        if abs(coeff) > 1e-10:
            rounded_coeff = round(coeff, precision)
            if rounded_coeff != 0:
                if i == 0:
                    terms.append(str(rounded_coeff))
                elif i == 1:
                    terms.append(f"{rounded_coeff}x")
                else:
                    terms.append(f"{rounded_coeff}x^{{{i}}}")
    polynomial = " + ".join(reversed(terms))
    polynomial = polynomial.replace("+ -", "- ")

    if len(polynomial) == 0:
        polynomial = '0'

    return polynomial


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Hermite polynomial interpolation"), width=12, className="text-center"),
        className="mb-5 mt-5"
    ),
    html.Div([
        dcc.Markdown('''For different nodes $x_0, x_1, ..., x_n$, for positive natural numbers $m_0, m_1, ..., m_n$ and known values $f^{(i)}(x_j)$, where $j=\{0, 1, ..., n\}$, $i = \{0, 1, ..., m_j-1\}$ we are looking for a polynomial $H=H(x)$ of degree $m$, where $m+1 = \sum_{j=0}^{n}m_j$, that satisfies the below condition:

$$
H_{(i)}(x_j) = f^{(i)}(x_j).
$$ 

Let the sought polynomial $H$ be given as
$$
H(x) = a_m x_m + a_{m-1} x^{m-1} + ... + a_1 x + a_0.
$$ 

Then, using the above conditions we obtain a system of $m+1$ linear equations with unknown quantities $a_0, a_1, ... a_m$. This system of equations has one possible solution. 

Here, we assume that for all nodes, 2 values are known, namely $y_j = f(x_j)$ and $z_j = f'(x_j)$.

''', mathjax=True)
    ]),
    dbc.Row(
        dbc.Col([
            dbc.Label("Number of nodes (n):", className="h4"),
            dbc.Input(id='num-nodes', type='number', min=1, max=100, step=1, value=3, className="form-control"),
            dbc.Tooltip("Provide the number of nodes", target='num-nodes'),
        ], width=4, className="offset-md-4"),
        className="mb-5"
    ),
    dbc.Row(id='input-rows', className="mb-2"),
    dbc.Row(
        dbc.Col(dbc.Button("Calculate", id='calculate-button', color="primary", className="mr-1"), width="auto"),
        className="justify-content-center mb-4"
    ),
    dbc.Row(
        dbc.Col(html.Div(id='output-container'), className="mb-4"),
    ),
    html.Div(id='polynomial-equation', className='mt-4', style={'fontSize': '20px'}),
    html.Div(id='error-container'),
    dbc.Row(
        dbc.Col(dcc.Graph(id='polynomial-plot', figure={}, mathjax=True), className="mb-5"),
    ),

    html.Div([
        html.Label("Provide the x value for which you want to calculate the value of the interpolated polynomial ", className="h4",
                   style={"display": "inline-block",
                          "margin-right": "10px", "fontSize": '16px'}),
        dbc.Input(id='x-to-calculate', type='text', value='0', className="form-control",  # type='number', value=0,
                  style={"display": "inline-block", "width": "100px"})
    ]),
    dbc.Row(
        dbc.Col(dbc.Button("Calculate", id='calculate-button-2', color="primary", className="mr-1"), width="auto"),
        className="justify-content-center mb-4"
    ),
    html.Div(id='H-calculated', className='mt-4', style={'fontSize': '20px'}),

    dcc.Store(id='nodes-store'),
    html.Script('''
        if (!window.dash_clientside) { window.dash_clientside = {}; }
        window.dash_clientside.clientside = {
            scrollToElements: function(trigger) {
                // Only run if trigger is truthy (to avoid running on initial load)
                if(trigger) {
                    const formulaElement = document.getElementById('polynomial-formula');
                    const plotElement = document.getElementById('polynomial-plot');
                    if(formulaElement) {
                        formulaElement.scrollIntoView({behavior: 'smooth', block: 'center'});
                    }
                    // Set a timeout to scroll to the plot after the formula
                    if(plotElement) {
                        setTimeout(function() {
                            plotElement.scrollIntoView({behavior: 'smooth', block: 'center'});
                        }, 500); // Adjust the timeout as needed
                    }
                }
                return window.dash_clientside.no_update; // Return no_update to avoid output errors
            }
        }
    '''),
    html.Div(id='dummy-output', style={'display': 'none'})
], fluid=True, className="py-3")


@app.callback(
    Output('input-rows', 'children'),
    Input('num-nodes', 'value')
)
def update_input_rows(num_nodes):
    return dbc.Row([
        dbc.Col([
            html.Div(f"Node {i + 1}:", className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText(dash_katex.DashKatex(expression=f"x_{i + 1}")),
                # dbc.InputGroupText(f"x{i + 1}"),
                dbc.Input(id={'type': 'dynamic-input', 'index': i}, type='text', className="mr-2",
                          style={'maxWidth': '100px'}, value='0'),
                dbc.Tooltip(f"Provide x for node {i + 1}", target={'type': 'dynamic-input', 'index': i}),
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText(dash_katex.DashKatex(expression=f"f(x_{i + 1})")),
                # dbc.InputGroupText(f"f(x{i + 1})"),
                dbc.Input(id={'type': 'dynamic-input', 'index': i + num_nodes}, type='text', className="mr-2",
                          style={'maxWidth': '100px'}, value='0'),
                dbc.Tooltip(f"Provide value f(x) for node {i + 1}",
                            target={'type': 'dynamic-input', 'index': i + num_nodes}),
            ], className="mb-2"),
            dbc.InputGroup([
                dbc.InputGroupText(dash_katex.DashKatex(expression=f"f'(x_{i + 1})")),
                # dbc.InputGroupText(f"f'(x{i + 1})"),
                dbc.Input(id={'type': 'dynamic-input', 'index': i + 2 * num_nodes}, type='text', className="mr-2",
                          style={'maxWidth': '100px'}, value='0'),
                dbc.Tooltip(f"Provide value f'(x) for node {i + 1}",
                            target={'type': 'dynamic-input', 'index': i + 2 * num_nodes}),
            ], className="mb-2"),
        ], width=2) for i in range(num_nodes)
    ], justify="center")


@app.callback(
    [Output('nodes-store', 'data'),
     Output('polynomial-plot', 'figure'),
     Output('polynomial-equation', 'children'),
     Output('error-container', 'children')],
    Input('calculate-button', 'n_clicks'),
    State('num-nodes', 'value'),
    [State({'type': 'dynamic-input', 'index': ALL}, 'value')],
    prevent_initial_call=True
)

def store_nodes_and_calculate(n_clicks, num_nodes, input_values_prev):

    input_values = []
    for element in input_values_prev:
        if '/' in element:
            try:
                element = float(element.split("/")[0]) / float(element.split("/")[1])
                input_values.append(element)
            except ValueError:
                return dash.no_update, {}, " ", dbc.Alert(
                    "Error: All cells must contain numbers.",
                    color="danger", dismissable=True, style={'fontSize': 20})
        else:
            try:
                input_values.append(float(element))
            except ValueError:
                return dash.no_update, {}, " ", dbc.Alert(
                    "Error: All cells must contain numbers.",
                    color="danger", dismissable=True, style={'fontSize': 20})

    nodes = [input_values[i:i + 3] for i in range(0, len(input_values), 3)]
    x_values = [node[0] for node in nodes]

    if len(x_values) != len(set(x_values)):
        return dash.no_update, {}, " ", dbc.Alert(
            "Error: Duplicated x values have been found. All x values must be unique. ", color="danger",
            dismissable=True, style={'fontSize': 20})

    wezly = np.array(nodes, dtype=float)
    try:
        coeffs = interpolation(wezly)
        delta = 0.5 * np.abs(max(x_values) - min(x_values))
        x_vals = np.linspace(min(x_values) - delta, max(x_values) + delta, 1000)
        y_vals = evaluate_polynomial(coeffs, x_vals)
        z_vals = evaluate_polynomial_derivative(coeffs, x_vals)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            legendgroup="first-degree",
            #legendgrouptitle_text="Polynomial",
            mode='lines',
            name=r'H(x)',
            line=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=z_vals,
            legendgroup="second-degree",
            #legendgrouptitle_text="Derivative",
            mode='lines',
            name=r"H'(x)",
            line=dict(color='yellowgreen')))
        fig.add_trace(go.Scatter(
            x=x_values,
            legendgroup="first-degree",
            y=[node[1] for node in nodes],
            mode='markers',
            marker=dict(color='navy', size=15),
            name=r"$(x_i, f(x_i))$"))
        fig.add_trace(go.Scatter(
            x=x_values,
            legendgroup="second-degree",
            y=[node[2] for node in nodes],
            mode='markers',
            marker=dict(color='olivedrab', size=15),
            name=r"$(x_i, f'(x_i))$"))
        # Calculate padding for x-axis and y-axis
        min_x_range = min(x_vals)
        max_x_range = max(x_vals)
        min_y_range = min(min(y_vals), min(z_vals))
        max_y_range = max(max(y_vals), max(z_vals))
        x_padding = 0.1 * np.abs(max_x_range - min_x_range)
        y_padding = 0.1 * np.abs(max_y_range - min_y_range)
        # Set the plot size dynamically based on values with padding
        fig.update_layout(
            legend_font_size=15,
            autosize=True,
            uirevision='constant',  # Keeps the plot settings consistent across updates
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                automargin=True,
                range=[min_x_range - x_padding, max_x_range + x_padding],
                fixedrange=False
            ),
            yaxis=dict(
                automargin=True,
                range=[min_y_range - y_padding, max_y_range + y_padding],
                fixedrange=False
            ),
            legend_grouptitlefont_size=15,
            legend_valign='bottom'
        )


        latex_poly = polynomial_to_latex(coeffs)
        return nodes, fig, dash_katex.DashKatex(
            expression=f"\\text{{Interpolated polynomial: H(x) = }} {latex_poly}"), " "
    except Exception as e:
        return dash.no_update, {}, dbc.Alert(e, color="danger", dismissable=True, style={'fontSize': 20}),


# Add a clientside callback to scroll to the elements
app.clientside_callback(
    """
    function(n_clicks) {
        if(n_clicks) {
            const formulaElement = document.getElementById('polynomial-equation');
            const plotElement = document.getElementById('polynomial-plot');
            if(formulaElement) {
                formulaElement.scrollIntoView({behavior: 'smooth', block: 'center'});
            }
            // Set a timeout to scroll to the plot after the formula
            setTimeout(function() {
                if(plotElement) {
                    plotElement.scrollIntoView({behavior: 'smooth', block: 'center'});
                }
            }, 1000); // Adjust the timeout as needed
        }
        return false;
    }
    """,
    Output('dummy-output', 'children'),
    [Input('calculate-button', 'n_clicks')],
    prevent_initial_call=True
)


@app.callback(
    Output('H-calculated', 'children'),
    Input('calculate-button-2', 'n_clicks'),
    [State('x-to-calculate', 'value'), State('nodes-store', 'data')],
    prevent_initial_call=True
)
def calculate_for_x(n_clicks, value_to_calculate, nodes):
    if n_clicks > 0:
        try:
            wezly = np.array(nodes, dtype=float)
            coeffs = interpolation(wezly)
        except IndexError:
            return dbc.Alert("Error: Nodes have not been provided.", color="danger", dismissable=True, style={'fontSize': 20})

        if '/' in value_to_calculate:
            try:
                value_to_calculate = float(value_to_calculate.split("/")[0]) / float(value_to_calculate.split("/")[1])
                evaluated = evaluate_polynomial(coeffs, value_to_calculate)
            except ValueError:
                return dbc.Alert("Error: The cell must contain a number.",
                                 color="danger", dismissable=True, style={'fontSize': 20})
        else:
            try:
                value_to_calculate = float(value_to_calculate)
                evaluated = evaluate_polynomial(coeffs, value_to_calculate)
            except ValueError:
                return dbc.Alert("Error: The cell must contain a number.",
                                 color="danger", dismissable=True, style={'fontSize': 20})
        return dash_katex.DashKatex(expression=f"\\text{{H({value_to_calculate}) = }} {evaluated}")


if __name__ == '__main__':
    app.run_server()
