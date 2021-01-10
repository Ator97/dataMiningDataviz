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
import plotly.graph_objects as go
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator       #https://github.com/arvkevi/kneed/blob/master/kneed/   Utiliza una interpolación



nut  = 1
ncd  = 1
ndm  = 1
nam  = 1
nc   = 1

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
        "   Decimal: ", dcc.Input(id='decimal', value='.', type='text'),
        html.Button('Cargar Archivo', id='loadFile', n_clicks=0,style={'width': '25%','margin': '3%'}),

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
                    value='pearson',style={'width': '50%','margin': '2%'}),
                html.Button('Ejecutar', id='executeCorr', n_clicks=0,style={'width': '25%','margin': '3%'}),
                dcc.Tabs(id="subtabs-1",value="subtab-1",children = [
                    dcc.Tab(label='Matriz de correlación',value='subtab-5',children=[
                        html.Div(id="crossMatrix"),]),                            
                    dcc.Tab(label='Grafica', value='subtab-2',children=[
                        html.Div(id="graphCrossMatrix"),]),
                ])
            ]),
            dcc.Tab(label='Apriori', value='tab-3',children = [
                html.Div([
                    "Soporte mínimo   ",        dcc.Input(
                id="soporteMinimo", type="number", placeholder="Valor de soporte mínimo",
                min=0, max=100, step=0.0001,value=0.003,style={'width': '6%','margin': '2%'}),
                    "     Confidencia mínima  ",        dcc.Input(
                id="confidenciaMinima", type="number", placeholder="Valor de confidencia mínimo",
                min=0, max=100, step=0.01,value=0.2,style={'width': '6%','margin': '2%'}),
                    "     Elevación mínima  ",        dcc.Input(
                id="elevacionMinima", type="number", placeholder="Valor de elevacion mínimo",
                min=0, max=100, step=0.01,value=3,style={'width': '6%','margin': '2%'}),
                    "     Tamaño mínimo  ",        dcc.Input(
                id="tamañoMinimo", type="number", placeholder="Valor de tamaño mínimo",
                min=0, max=100, step=0.01,value=2,style={'width': '6%','margin': '2%'}),
                html.Button('     Ejecutar', id='executeAprori', n_clicks=0,style={'margin': '2%'}),
                html.Div(id="aprioriMatrix")
                ])
            ]),
            dcc.Tab(label='Distancias', value='tab-4',children=[
               dcc.Dropdown(
                    id='distance',
                    options=[
                        {'label': 'Chebyshev', 'value': 'chebyshev'},
                        {'label': 'Cityblock', 'value': 'cityblock'},
                        {'label': 'Euclidean', 'value': 'euclidean'}
                    ],
                    value='euclidean',style={'width': '50%','margin': '2%'}),
                html.Button('Ejecutar', id='executeDis', n_clicks=0,style={'width': '25%','margin': '3%'}),
                html.Div(id="distanceMatrix"),                          
            ]),
            dcc.Tab(label='Clustering Jerarquico', value='tab-5',children=[

                html.Button('Ejecutar', id='executeCluster', n_clicks=0,style={'width': '25%','margin': '3%'}),
                    dcc.Tabs(id="subtabs-2",value="subtab-2",children = [
                        dcc.Tab(label='Grafica del Codo',value='subtab-5',children=[
                            html.Div(id="elbow"),
                        ]),  
                        dcc.Tab(label='Gráfica del Cluster', value='subtab-2',children=[
                            html.Div(id="cluster"),
                        ]),
                ])
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
        Input("decimal","value"),
        Input("loadFile", "n_clicks")
    ]
)
def update_table(contents, filename,separador,decimal,n_clicks):
    table = html.Div()
    global nut
    if nut == n_clicks:
        nut = nut +1
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
        Input('correlationMethod','value'),
        Input('executeCorr', 'n_clicks')
    ]
)
def crossData(decimal,separador,contents, filename,correlationMethod,n_clicks):
    table = html.Div()
    figure = dcc.Graph()
    global ncd
    if ncd == n_clicks:
        if contents:
            ncd = ncd + 1
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
        Input('distance','value'),
        Input('executeDis', 'n_clicks') 
    ]
)
def distanceMatrix(decimal,separador,contents, filename,correlationMethod,n_clicks):
    table = html.Div()
    global ndm
    if ndm == n_clicks:
        if contents:
            ndm = ndm + 1
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


def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

@app.callback(
    Output('aprioriMatrix', 'children'),
    [
        Input('decimal','value'),
        Input('separador','value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('soporteMinimo','value'),
        Input('confidenciaMinima','value'),
        Input('elevacionMinima','value'),
        Input('tamañoMinimo','value'),
        Input('executeAprori', 'n_clicks')
    ]
)
def aprioriMatrix(decimal,separador,contents, filename,soporteMinimo,confidenciaMinima,elevacionMinima,tamañoMinimo,n_clicks):
    table = html.Div()
    global nam
    if nam == n_clicks:
        if contents :
            nam = nam+1
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)
            df = df.set_index(df.columns[0])

            transactions = []
            for i in range(0, len(df.index)):
                transactions.append([str(df.values[i,j]) for j in range(0, len(df.columns) )])

            # Training Apriori on the dataset
            from apyori import apriori
            rules = apriori(transactions, min_support = soporteMinimo, min_confidence = confidenciaMinima,  min_lift = elevacionMinima, min_length = tamañoMinimo)

            # Resultados
            results = list(rules)

            # Este comamdo crea un frame para ver los datos resultados
            df=pd.DataFrame(inspect(results),
                            columns=['rhs','lhs','Soporte','Confidencia','Elevación'])


            table = html.Div([
                dash_table.DataTable(
                    data=df.to_dict("rows"),
                    columns=[{"name": str(i), "id": str(i)} for i in df.columns],
                    fixed_rows={'headers': True},
                    style_table={'overflowX': 'auto','overflowY': 'auto'},
                    style_cell={
                    'minWidth': '180px', 'width': '180px', 'maxWidth': '180px',
                    'overflow': 'scroll'  }
                ), 
            ])

    return table


@app.callback(
    [
    Output('elbow', 'children'),
    Output('cluster','children')
    ],[
        Input('decimal','value'),
        Input('separador','value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('executeCluster', 'n_clicks')

    ]
)
def clustering(decimal,separador,contents, filename,n_clicks):

    figure1 = dcc.Graph()
    figure2 = dcc.Graph()
    global nc
    if nc == n_clicks:
        if contents :
            nc = nc+1
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)
            VariablesModelo = df.iloc[:,:].values
            SSE = []
            for i in range(2, 16):
                km = KMeans(n_clusters=i)
                km.fit(VariablesModelo)
                SSE.append(km.inertia_)

            x = np.arange(len(SSE))
            fig = go.Figure( data=   go.Scatter(x=x,y=SSE))

            kl = KneeLocator(range(2, 16), SSE, curve="convex", direction="decreasing")
            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(VariablesModelo)

            model = KMeans(n_clusters = kl.elbow, init = "k-means++", max_iter = 300, n_init = 10, random_state =   0)
            y_clusters = model.fit_predict(VariablesModelo)


            labels = model.labels_
            trace = go.Scatter3d(x=VariablesModelo[:, 0], y=VariablesModelo[:, 1], z=VariablesModelo[:, 2],     mode='markers',marker=dict(color = labels, size= 3,      line=dict(color= 'black',width = 3)))
            layout = go.Layout(margin=dict(l=0,r=0))
            data = [trace]
            fig2 = go.Figure(data = data, layout = layout)

            figure1 = html.Div(
                [
                    dcc.Graph(
                        id='kind',
                        figure=fig
                    ),
                ]
            )

            figure2 = html.Div(
                [
                    dcc.Graph(
                        id='kind2',
                        figure=fig2
                    ),
                ]
            )

    return figure1,figure2



if __name__ == '__main__':
    app.run_server(debug=True)
