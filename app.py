import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table


import pandas as pd

import plotly.express as px

from apyori import apriori
from scipy.spatial import distance


df = pd.read_csv('./1Cancer.csv')

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



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H6("Data Minning Crawler"),

    dcc.Upload( 
        id='upload-data',
        children=html.Div(['Toma y suelta o ', html.A('seleciona el archivo')]),
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
            multiple=True
    ),

    html.Div([
        "Separador: ", dcc.Input(id='separador', value=',', type='text'),
        "   Decimal: ", dcc.Input(id='decimal', value='.', type='text') 
        ]),    

    dcc.Tabs(id='tabsControlInput', value='tab-1', 
        children=[
            dcc.Tab(label='Set de datos', value='tab-1',children=[
                html.Div(id="output-data-upload"),
            ]),
            dcc.Tab(label='Correlación', value='tab-2',children=[
                dcc.Dropdown(
                    id='correlationMethod',
                    options=[
                        {'label': 'Pearson', 'value': 'pearson'},
                        {'label': 'Kendall', 'value': 'kendall'},
                        {'label': 'Spearman', 'value': 'spearman'}
                    ],
                    value='pearson'),
                dcc.Tabs(id="subtabs",value="subtab-1",children = [
                    dcc.Tab(label='Matriz de correlación',value='subtab-5',children=[
                        html.Div(id="crossMatrix"),]),                            
                    dcc.Tab(label='Grafica', value='subtab-2',children=[
                        html.Div(id="graphCrossMatrix"),]),
                ])
            ]),
            dcc.Tab(label='Apryori', value='tab-3',children = [
                dcc.Tabs(id="subtabs2",value="subtab-3",children = [
                    dcc.Tab(label='Grafica cuadrangular',value='subtab-3',children=[
                        print("D")
                        #dcc.Graph(
                        #id='basic-interactions3',figure=fig3)
                    ]),
                    dcc.Tab(label='Grafica cuadrangular',value='subtab-4',children=[
                        print("D")
                        #dcc.Graph(
                        #    id='basic-interactions4',figure=fig)
                    ]),
                ]),
            ]),
            dcc.Tab(label='Distancias', value='tab-4',children=[
               dcc.Dropdown(
                    id='distance',
                    options=[
                        {'label': 'Chebyshev', 'value': 'chebyshev'},
                        {'label': 'Cityblock', 'value': 'cityblock'},
                        {'label': 'Euclidean', 'value': 'euclidean'}
                    ],
                    value='euclidean'),
                html.Div(id="distanceMatrix"),                          
            ]),
        ]
    ),

    html.Div(id='tabsControl'),
    html.Div(id='subtabsControl')

])


def parse_data(contents, filename,separador,decimal):
    content_type, content_string = contents.split(separador)

    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")),decimal=decimal)
        elif "xls" in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif "txt" or "tsv" in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+")
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])

    return df


@app.callback(
    Output("output-data-upload", "children"),
    [
        Input("upload-data", "contents"), 
        Input("upload-data", "filename"),
        Input("separador","value"),
        Input("decimal","value")
    
    ]
)
def update_table(contents, filename,separador,decimal):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename,separador,decimal)

        table = html.Div(
            [
                html.H5(filename),
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    fixed_rows={'headers': True},

                ),
                
                
            ]
        )

    return table


@app.callback(
    [
        Output('graphCrossMatrix','children') ,
        Output('crossMatrix', 'children'),
    ],[
        Input('decimal','value'),
        Input('separador','value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('correlationMethod','value')
    ]
)
def crossData(decimal,separador,contents, filename,correlationMethod):
    table = html.Div()
    figure = dcc.Graph()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename,separador,decimal)
        df = df.set_index(df.columns[0])
        df = df.corr(method=correlationMethod)
        table = html.Div(
            [
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": i, "id": i} for i in df.columns],
                ), 
            ]
        ),
        fig = px.imshow(df)
        figure = html.Div(
            [
                dcc.Graph(
                    id='kind',
                    figure=fig
                ),
            ]
        )

    return figure,table
 

@app.callback(
    Output('distanceMatrix', 'children'),
    [
        Input('decimal','value'),
        Input('separador','value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('distance','value')
    ]
)
def crossData(decimal,separador,contents, filename,correlationMethod):
    table = html.Div()

    if contents:
        contents = contents[0]
        filename = filename[0]
        df = parse_data(contents, filename,separador,decimal)
        df = df.set_index(df.columns[0])

        index = df.index[:].tolist()
        df = df.values.tolist()
        df= [df[i] + [index[i]]  for i in range(0,len(df))]

        l= []
        for i in df:
            ll=[]
            for j in df:
                if correlationMethod == 'euclidean':
                    ll.append(round(distance.euclidean(i, j),2))
                elif correlationMethod == 'cityblock':
                    ll.append(round(distance.cityblock(i, j),2))
                elif correlationMethod == 'chebyshev':
                    ll.append(round(distance.chebyshev(i, j),2))
            l.append(ll)

        df = pd.DataFrame(l)
        print(df)
        table = html.Div(
            [
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": str(i), "id": str(i),"type":"numeric"} for i in df.columns],
                    fixed_rows={'headers': True},
                    style_table={'overflowX': 'auto','overflowY': 'auto'},
                    style_cell={
                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'overflow': 'scroll'  }
                ), 
            ]
        )

    return table




if __name__ == '__main__':
    app.run_server(debug=True)
