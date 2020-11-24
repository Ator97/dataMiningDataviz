import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')

df = pd.DataFrame({
    "x": [1,2,1,2],
    "y": [1,2,3,4],
    "customdata": [1,2,3,4],
    "fruit": ["apple", "apple", "orange", "orange"]
})
fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=20)

fig2 = px.imshow([[1, 20, 30],
                 [20, 1, 60],
                 [30, 60, 1]])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H6("Data Minning Crawler"),

    html.Div([
        dcc.Upload( 
        id='upload-data',
        children=html.Div([
            'Toma y suelta o ',
            html.A('seleciona el archivo')
        ]),
        style={
            'width': '98%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
    ),

    html.Div([
        "Separador: ", dcc.Input(id='separador', value=',', type='text'),
        "   Decimal: ", dcc.Input(id='decimal', value='.', type='text'),
        dcc.Checklist(
            options=[ {'label': 'Header', 'value': 'Yes'}],
            value=['Yes', 'No']) 
        ]
    ),
    ]),

    dcc.Tabs(id='tabsControlInput', value='tab-1', 
        children=[
            dcc.Tab(label='Set de datos', value='tab-1'),
            dcc.Tab(label='Correlación', value='tab-2',
                children = [
                    dcc.Tabs(id="subtabs",value="subtab-1",
                        children = [
                            dcc.Tab(label='Analisis gráfico', value='subtab-1'),
                            dcc.Tab(label='Grafica', value='subtab-2'),
                        ])]),
            dcc.Tab(label='Apryori', value='tab-3'),
            dcc.Tab(label='Distancias', value='tab-4'),
        ]
    ),
    
    html.Div(id='tabsControl'),
    html.Div(id='subtabsControl')

])


@app.callback(Output('tabsControl', 'children'),
              Input('tabsControlInput', 'value'),
              )
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
            )
        ])
    elif tab == 'tab-2':
        pass

    elif tab == 'tab-3':
        return html.Div([
            html.H3('Apriory')
        ])
    elif tab == 'tab-4':
        return html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
            )
        ])


@app.callback(Output('subtabsControl', 'children'),
              Input('subtabs', 'value'),
              )
def render_content2(tab):
    if tab == 'subtab-1':
        return html.Div([
            dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),
    ])
    elif tab == 'subtab-2':
        return html.Div([
            dcc.Graph(
        id='basic-interactions',
        figure=fig2
    ),
    ])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    app.run_server(debug=True)