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
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator       

# Variables de control
nut  = 1 #Variable de contencion para actualizacion de datos cargados
ncd  = 1 #Varaible de contencion para ejecucion de matriz cruzada
ndm  = 1 #Varaible de contencion para ejecucion de matriz de distancias
nam  = 1 #Varaible de contencion para ejecucion de apriori
nc   = 1 #Varaible de contencion para ejecucion de clasificacion por clustering
ns   = 1 #Varaible de contencion para ejecucuion de clasificacion sigmoide
mensaje = "" #Varaible para mostrar mensaje del resultado sigmoide


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#Interfaz del sistema
app.layout = html.Div([
    #Titulo
    html.H6("Data Minning Crawler"),
    #Menu de configuracion para cargar archivos
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
            # Por si queremos analizar mas archivos en la misma sesion
            multiple=True
    ),
    html.Div([
        "Separador: ", dcc.Input(id='separador', value=',', type='text'),
        "   Decimal: ", dcc.Input(id='decimal', value='.', type='text'),
        html.Button('Cargar Archivo', id='loadFile', n_clicks=0,style={'width': '25%','margin': '3%'}),

        ]),    
    #Seccion de pestañas 
    dcc.Tabs(id='tabsControlInput', value='tab-1', 
        children=[
            #Pestaña con datos cargados
            dcc.Tab(label='Set de datos', value='tab-1',children=[
                html.Div(id="output-data-upload"),
            ]),
            #Pestalla con matriz de correlacion y grafica de correlacion
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
            #Pestaña con resultados de algoritmo apriori
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
            #Pestaña con resultados de matriz de distancias
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
            #Pestaña con resultado de clustering particional
            dcc.Tab(label='Clustering Particional', value='tab-5',children=[

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
            #Pestaña de menu de dteccion de cancer. Solicitada en expreso
            dcc.Tab(label='Clasificación Sigmoide', value='tab-6',children = [
                html.Button('     Ejecutar', id='executeSigmoide', n_clicks=0,
                    style={'margin': '2%','textAlign': 'center'}),

                html.Div([
                    html.Div([
                        "Compactividad",
                        dcc.Input(
                            id="compactividad",  type="number", value= 0.04362,
                            placeholder="Compactividad",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                    html.Div([
                        "Textura",    
                        dcc.Input(
                            id="textura",     type="number", value=24.54,
                            placeholder="Textura",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                ],className="row"),
                html.Div([
                    html.Div([
                        "Area",       
                        dcc.Input(
                            id="area",        type="number", value=181.0,
                            placeholder="Area",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                    html.Div([
                        "Concavidad", 
                        dcc.Input(
                            id="concavidad",  type="number",value = 0, 
                            placeholder="Concavidad",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                ],className="row"),

                html.Div([
                    html.Div([
                        "Simetria",   
                        dcc.Input(
                            id="simetria",  type="number", value=0.1587,
                            placeholder="Simetria",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                    html.Div([
                        "Dimensión fractal", 
                        dcc.Input(
                            id="dimensionFractal",type="number", value=1.0,
                            placeholder="Dimensión Fractal",style={'margin': '5%','textAlign': 'center'}),
                    ],className="six columns"),
                ],className="row"),

                    html.Div(id='sigmoide',style={'textAlign': 'center'}),

            ]),
        ]),
    #Cargad de los anterior contruido
    html.Div(id='tabsControl'),
    html.Div(id='subtabsControl')
])

#Funcion de carga
# Extraida directamente de la documetnacion de Dash.com
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

#Prodcimiento de carga de archivos, extraido de la documentacion de Dash.com
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

#Forma ejecucion de analisis para matriz cruzada
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
    #Objetos a retornar
    table = html.Div()
    figure = dcc.Graph()
    #Ejecutamos despues de asegurar datos de entrada
    global ncd
    if ncd == n_clicks:
        if contents:
            ncd = ncd + 1
            contents = contents[0]
            filename = filename[0]
            #Calculamos matriz de correlacion con ayuda de valor extrarno llamado correlationMethod
            df = parse_data(contents, filename,separador,decimal)
            df = df.set_index(df.columns[0])
            df = df.corr(method=correlationMethod)
            #Retornamos el objeto tabla
            table = html.Div(
                [
                    dash_table.DataTable(
                        data=df.to_dict("rows"),
                        columns=[{"name": i, "id": i} for i in df.columns],
                    ), 
                ]
            ),
            #Retornamos el objeto grafica de los resultados basados en la tabla
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
 
#Forma de ejecucion de analisi para matriz de distancias
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
    #Objeto a retornar
    table = html.Div()
    #Empezamos analisis
    global ndm
    if ndm == n_clicks:
        if contents:
            ndm = ndm + 1
            #Cargamos datos
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)
            df = df.set_index(df.columns[0])
            index = df.index[:].tolist()
            df = df.values.tolist()
            df= [df[i] + [index[i]]  for i in range(0,len(df))]

            l= []
            #Realizamos analis
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
            #Formateamos datos por estilo
            df = pd.DataFrame(l)
            #objeto tabla a retornar.
            #Apadtativo por la cantidad de columnas probables
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

#Forma elegante de generar  tabla que pose los valores de un analisis apriori.
#Variblas a retornar verbosas
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))

#Forma de ejecucion de analissi para matriz de distancias
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
    #Objeto a retornar
    table = html.Div()
    #Comenzamos analisis
    global nam
    if nam == n_clicks:
        if contents :
            nam = nam+1
            #Cargamos datos
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)
            df = df.set_index(df.columns[0])

            #La primer columna es tomada como indice, reverimos eso
            transactions = []
            for i in range(0, len(df.index)):
                transactions.append([str(df.values[i,j]) for j in range(0, len(df.columns) )])

            #Entremamos algoritmo
            from apyori import apriori
            rules = apriori(transactions, min_support = soporteMinimo, min_confidence = confidenciaMinima,  min_lift = elevacionMinima, min_length = tamañoMinimo)

            # Resultados
            results = list(rules)

            # Este comamdo crea un frame para ver los datos resultados
            df=pd.DataFrame(inspect(results),
                            columns=['rhs','lhs','Soporte','Confidencia','Elevación'])

            #Objeto a retornar
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

#Forma de ejecicion de analisis para clustering
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
    #Objetos a retornar
    figure1 = dcc.Graph()
    figure2 = dcc.Graph()
    #Comenzamos analisis
    global nc
    if nc == n_clicks:
        if contents :
            nc = nc+1
            #Cargamos datos
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)

            #OBtenemos variables modelo
            VariablesModelo = df.iloc[:,:].values
            SSE = []
            for i in range(2, 16):
                km = KMeans(n_clusters=i)
                km.fit(VariablesModelo)
                SSE.append(km.inertia_)
            #Obtenemos numero de clusters con grafica de codo
            x = np.arange(len(SSE))
            fig = go.Figure( data=   go.Scatter(x=x,y=SSE))
            #Obtenemos la canditdad optima de clusters
            kl = KneeLocator(range(2, 16), SSE, curve="convex", direction="decreasing")
            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(VariablesModelo)
            #Generamos el modelo segun la catndidad de clusters previamente calculada
            model = KMeans(n_clusters = kl.elbow, init = "k-means++", max_iter = 300, n_init = 10, random_state =   0)
            y_clusters = model.fit_predict(VariablesModelo)

            #Obenemos el comportamiento de los datos en una grafica de tres dimensiones
            labels = model.labels_
            trace = go.Scatter3d(x=VariablesModelo[:, 0], y=VariablesModelo[:, 1], z=VariablesModelo[:, 2],     mode='markers',marker=dict(color = labels, size= 3,      line=dict(color= 'black',width = 3)))
            layout = go.Layout(margin=dict(l=0,r=0))
            data = [trace]
            fig2 = go.Figure(data = data, layout = layout)
            #Grafica de codo a retornar
            figure1 = html.Div(
                [
                    dcc.Graph(
                        id='kind',
                        figure=fig
                    ),
                ]
            )
            #Gracia de clusters a retornar
            figure2 = html.Div(
                [
                    dcc.Graph(
                        id='kind2',
                        figure=fig2
                    ),
                ]
            )

    return figure1,figure2

#Forma de ejecucion de analisis sigmoide
@app.callback(
    Output('sigmoide', 'children'),
    [
        Input('compactividad','value'),
        Input('textura','value'),
        Input('area','value'),
        Input('concavidad','value'),
        Input('simetria','value'),
        Input('dimensionFractal','value'),
        Input('decimal','value'),
        Input('separador','value'),
        Input('upload-data', 'contents'),
        Input('upload-data', 'filename'),
        Input('executeSigmoide', 'n_clicks')
    ]
)
def sigmoide(compactividad,textura,area,concavidad,simetria,dimensionFractal,decimal,separador,contents, filename,n_clicks):
    #Objeto a retornar
    mensaje = html.Div()
    #Empezamos analisis
    global ns
    if ns == n_clicks:
        if contents :
            ns = nc+1
            #Cargamos los datos
            contents = contents[0]
            filename = filename[0]
            df = parse_data(contents, filename,separador,decimal)
            df = df.set_index(df.columns[0])
            #Obtenemos las caracteristicas  principales de forma estatica.
            X = np.array(df[['Texture', 'Area', 'Compactness','Concavity', 'Symmetry', 'FractalDimension']])
            Y = np.array(df[['Diagnosis']])

            #Preparativos del modelo
            Clasificacion = linear_model.LogisticRegression()
            validation_size = 0.2
            seed = 1234
            #Variables a usar como entrenamiento y validacion
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
                                X, Y, test_size=validation_size, random_state=seed, shuffle = True)
            Clasificacion.fit(X_train, Y_train)
            Probabilidad = Clasificacion.predict_proba(X_train)
            Predicciones = Clasificacion.predict(X_train)
            Clasificacion.score(X_train, Y_train)
            #Prediccion nueva segun los datos conocidos
            PrediccionesNuevas = Clasificacion.predict(X_validation)
            confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, 
                                rownames=['Real'], colnames=           ['Predicción'])

            v = Clasificacion.score(X_validation, Y_validation)

            NuevoPaciente = pd.DataFrame({  'Texture': [textura],           'Area': [area], 
                                            'Compactness': [compactividad], 'Concavity': [concavidad], 
                                            'Symmetry': [simetria],         'FractalDimension': [dimensionFractal]})
            
            print(Clasificacion.predict(NuevoPaciente))
            #Retornamos la prediccion con el grado de certeza
            if  Clasificacion.predict(NuevoPaciente) == "B":
                mensaje = html.Div(
                        html.H5("Con una  certeza del " + str(format(v*100, '.2f') ) +"% se pronostica POSITIVO a Cancer ")
                )
            else:
                mensaje = html.Div(
                        html.H5("Con una  certeza del " + str(format(v*100, '.2f'))  +"% se pronostica NEGATIVO a Cancer "))

    return mensaje

if __name__ == '__main__':
    app.run_server(debug=True)
