import base64
import datetime
import io

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px

from apyori import apriori
from math import sqrt
from scipy.spatial import distance

import pandas as pd
df = pd.read_csv('./1Cancer.csv')


Matriz = df.corr(method='pearson')
fig = px.scatter(df)
fig.update_layout(clickmode='event+select')
fig.update_traces(marker_size=20)
fig2 = px.imshow(Matriz)




l= []
for i in df.iloc:
    ll=[]
    for j in df.iloc:
        ll.append(distance.euclidean(i, j))
    l.append(ll)

regla = []
soporte = []
confianza = []
Reglas = apriori(df, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2) 
for item in Reglas:
    pair = item[0]
    items = [x for x in pair]
    soporte.append(item[1])
    confianza.append(item[2][0][2])
    regla.append(items[0] + " -> " + items[1])
df2 = pd.DataFrame({
    "support": soporte,
    "confidence": confianza,
    "customdata": regla,
})


fig3 = px.scatter(df2, x="support", y="confidence", custom_data=["customdata"])
fig3.update_layout(clickmode='event+select')
fig3.update_traces(marker_size=20)


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
            dcc.Tab(label='Set de datos', value='tab-1',children=[
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in df.columns],
                                data=df.to_dict('records'),    
                                fixed_rows={'headers': True},
                                style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},)
            ]),
            dcc.Tab(label='Correlación', value='tab-2',
                children = [
                    dcc.Tabs(id="subtabs",value="subtab-1",
                        children = [
                            dcc.Tab(label='Matriz de correlación',
                                value='subtab-5',children=[
                                dash_table.DataTable(
                                    id='table2',
                                    columns=[{"name": i, "id": i} for i in Matriz.columns],
                                    data=Matriz.to_dict('records'),
                                    fixed_rows={'headers': True},
                                    style_cell={'minWidth': '100px', 'width': '100px', 'maxWidth': '100px'},)
                                ]),                            
                            dcc.Tab(label='Grafica', value='subtab-2',
                                children=[dcc.Graph(id='basic-interactions2',
                                figure=fig2),]),])]),
            
            dcc.Tab(label='Apryori', value='tab-3',
                children = [
                    dcc.Tabs(id="subtabs2",value="subtab-3",
                        children = [
                            dcc.Tab(label='Grafica cuadrangular',
                                value='subtab-3',children=[dcc.Graph(
                                id='basic-interactions3',figure=fig3)]),
                            dcc.Tab(label='Grafica cuadrangular',
                                    value='subtab-4',children=[dcc.Graph(
                                    id='basic-interactions4',figure=fig)]),
                            ]),]),
            dcc.Tab(label='Distancias', value='tab-4',children=[
                            dash_table.DataTable(
                id='table3',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'))
            ]),
        ]
    ),
    
    html.Div(id='tabsControl'),
    html.Div(id='subtabsControl')

])




#@app.callback(Output('subtabsControl', 'children'),
#              Input('subtabs', 'value'),
#              )
#def render_content2(tab):
#    if tab == 'subtab-1':
#        return html.Div([
#            dcc.Graph(
#        id='basic-interactions',
#        figure=fig
#    ),
#    ])
#    elif tab == 'subtab-2':
#        return html.Div([
#            dcc.Graph(
#        id='basic-interactions2',
#        figure=fig2
#    ),
#    ])


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
