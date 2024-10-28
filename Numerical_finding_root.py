import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
import decimal

#http://127.0.0.1:8050/
def evaluate_polynomial(coeffs, x_values):
    result = np.polyval(coeffs, x_values)
    return result

def find_decimals(value):
    return (np.abs(decimal.Decimal(str(value)).as_tuple().exponent))


def Newton(x0, epsilon):
    if (epsilon >= 1) or (epsilon <= 0):
        pass

    else:
        try:
            precision = find_decimals(epsilon)
            coeffs = [1, 1, 0, 7]
            coeffs_denominator = [0, 3, 2, 0]
            coeffs_derivative_x = [3, 2, 0, 0]
            coeffs_numerator = np.array(coeffs_derivative_x) - np.array(coeffs)
            evaluated_numerator = evaluate_polynomial(coeffs_numerator, x0)
            evaluated_denominator = evaluate_polynomial(coeffs_denominator, x0)
            xn = evaluated_numerator / evaluated_denominator
            steps = 1

            while np.abs(evaluate_polynomial(coeffs, xn)) > epsilon:
                evaluated_numerator = evaluate_polynomial(coeffs_numerator, xn)
                evaluated_denominator = evaluate_polynomial(coeffs_denominator, xn)
                xn = evaluated_numerator / evaluated_denominator
                steps += 1

            return round(xn, precision), steps  # zwraca wartość xn zaokrągloną do liczby cyfr po przecinku epsilona
        except Exception as e:
            print(e)

def bisection(a, b, epsilon):
    coeffs = [1, 1, 0, 7]
    f_a = evaluate_polynomial(coeffs, a)
    f_b = evaluate_polynomial(coeffs, b)

    if f_a * f_b >= 0:
        pass


    elif (epsilon < 0) or (epsilon > 1):
        pass

    else:
        try:
            precision = find_decimals(epsilon)
            steps = 0
            c = (a + b) / 2

            while np.abs(evaluate_polynomial(coeffs, c)) > epsilon:
                if evaluate_polynomial(coeffs, a) * evaluate_polynomial(coeffs, c) < 0:
                    a = a
                    b = c
                else:
                    a = c
                    b = b

                c = (a + b) / 2
                steps += 1

            return round(c, precision), steps
        except Exception as e:
            print(e)

# X range for the plot
x_range = np.linspace(-5, 5, 1000)  # Adjust range and density as needed
x_min, x_max = min(x_range), max(x_range)
y_values = evaluate_polynomial([1, 1, 0, 7], x_range)

# Polynomial plot
poly_trace = go.Scatter(x=x_range, y=y_values, mode='lines', name='Polynomial')

# Layout including the range slider
layout = go.Layout(
    title="",
    xaxis=dict(
        title='x',
        rangeslider=None,
        range=[x_min, x_max],  # Fix the range of the x-axis
        type="linear"
    ),
    yaxis=dict(title='y')
)
# Initialize Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

valid_style = {'className': 'mb-2'}
invalid_style = {'className': 'mb-2', 'style': {'borderColor': 'red', 'backgroundColor': '#FFEEEE'}}

app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("Newton's and Bisection method of finding a root of a function"), width=12, className="text-center"),
        className="mb-5 mt-5"
    ),
    dbc.Row(
        dbc.Col([
            dcc.Markdown('''
                $$
                x^3 + x^2 + 7 = 0
                $$
                ''', mathjax=True)
        ], width=8, className="offset-md-2"),
        className="text-center"
    ),
    dbc.Row(
        dcc.Graph(id='poly-plot', figure={'data': [poly_trace], 'layout': layout})
    ),
    dbc.Row([
        # Column for Newton's Method
        dbc.Col([
            html.H3("Newton's method", className="text-center"),
            dbc.Input(id='x0-input', type='number', placeholder="Provide the initial value", className="mb-2"),
            dbc.Input(
                id='epsilon-input-newton',
                type='number',
                min=0,
                max=1,
                step='any',
                placeholder="Provide the truncation error of the method (epsilon)",
                **valid_style
            ),
            dbc.Button('Calculate', id='newton-button', color='primary', className="mb-2"),
            html.Div(id='newton-output', className="text-center")
        ], width={"size": 4, "offset": 1}, className="mb-4"),  # Adjusted width and offset
        # Column for Bisection Method
        dbc.Col([
            html.H3("Bisection method", className="text-center"),
            dbc.Row([
                dbc.Col(dbc.Input(id='a-input', type='number', placeholder="Provide a"), width=3),
                dbc.Col(dcc.RangeSlider(
                    id='ab-slider',
                    min=-5,
                    max=5,
                    step=0.1,
                    marks={i: str(i) for i in range(-5, 6)},
                    value=[-5, 5],
                    className="mb-2",
                    allowCross=False,
                    tooltip={"placement": "bottom", "always_visible": True}
                ), width=6),
                dbc.Col(dbc.Input(id='b-input', type='number', placeholder="Provide b"), width=3),
            ], justify="center"),
            dbc.Input(
                id='epsilon-input-bisekcja',
                type='number',
                step='any',
                min=0,
                max=1,
                placeholder="Provide the truncation error of the method (epsilon)",
                **valid_style
            ),
            dbc.Button('Calculate', id='bisekcja-button', color='primary', className="mb-3"),
            html.Div(id='bisekcja-output', className="text-center")
        ], width={"size": 4}, className="mb-4"),  # Adjusted width
    ], className="mb-4", justify="center"),
], fluid=True, className="py-3")


@app.callback(
    [Output('a-input', 'value'), Output('b-input', 'value')],
    [Input('ab-slider', 'value')]
)
def update_inputs_from_slider(slider_value):
    if slider_value is not None:
        a_value, b_value = slider_value
        return a_value, b_value
    else:
        return dash.no_update, dash.no_update

# Optionally, a callback to update the graph's slider based on input fields
@app.callback(
    Output('a-input', 'style'),
    [Input('b-input', 'value'), Input('a-input', 'value')]
)
def validate_a_input(b_value, a_value):
    if b_value is not None and a_value is not None and a_value >= b_value:
        return invalid_style['style']
    return valid_style.get('style', {})

@app.callback(
    Output('b-input', 'style'),
    [Input('a-input', 'value'), Input('b-input', 'value')]
)
def validate_b_input(a_value, b_value):
    if a_value is not None and b_value is not None and b_value <= a_value:
        return invalid_style['style']
    return valid_style.get('style', {})


# Callback for Newton's method
@app.callback(
    Output('newton-output', 'children'),
    [Input('newton-button', 'n_clicks')],
    [State('x0-input', 'value'), State('epsilon-input-newton', 'value')]
)
def update_newton_output(n_clicks, x0, epsilon):
    if n_clicks and n_clicks > 0:  # Ensures that n_clicks is not None and greater than 0
        if None not in (x0, epsilon):  # Check if any of the inputs are None
            try:
                if epsilon >= 1:
                    return "The truncation error must be a number from the range (0,1)."
                elif epsilon <= 0:
                    return "The truncation error must be a number from the range (0,1)."
                else:
                    x0 = float(x0)
                    epsilon = float(epsilon)
                    result, steps = Newton(x0, epsilon)
                    return f"Result: {result}, Number of steps: {steps}"
            except (TypeError, ValueError) as e:
                return f"Error: {str(e)}"  # Return the error message to the user
        else:
            if x0 is None:
                return "Provide the initial value."
            if epsilon is None:
                return "The truncation error must be a number from the range (0,1)."

    return ""

@app.callback(
    Output('bisekcja-output', 'children'),
    [Input('bisekcja-button', 'n_clicks')],
    [State('a-input', 'value'), State('b-input', 'value'), State('epsilon-input-bisekcja', 'value')]
)
def update_bisekcja_output(n_clicks, a, b, epsilon):
    if n_clicks and n_clicks > 0:  # Ensures that n_clicks is not None and greater than 0
        if None not in (a, b, epsilon):  # Check if any of the inputs are None
            try:
                if epsilon >= 1:
                    return "The truncation error must be a number from the range (0,1)."
                elif epsilon <= 0:
                    return "The truncation error must be a number from the range (0,1)."
                else:
                    a = float(a)
                    b = float(b)
                    coeffs = [1, 1, 0, 7]
                    f_a = evaluate_polynomial(coeffs, a)
                    f_b = evaluate_polynomial(coeffs, b)

                    epsilon = float(epsilon)

                    if f_a * f_b >= 0:
                        return "Please provide appropriate values. The values must be of different signs (one positive, one negative)."
                    else:
                        result, steps = bisection(a, b, epsilon)
                        return f"Result: {result}, Number of steps: {steps}"
            except (TypeError, ValueError) as e:
                return f"Error: {str(e)}"  # Return the error message to the user
        else:
            if a >= b:
                return "Please provide appropriate values. Assuming interval (a,b) a must be less than b."
            if a is None:
                return "Please provide the initial values."
            if b is None:
                return "Please provide the initial values."
            if epsilon is None:
                return "The truncation error must be a number from the range (0,1)."

    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

