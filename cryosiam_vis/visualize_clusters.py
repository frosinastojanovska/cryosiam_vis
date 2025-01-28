import io
import os
import math
import yaml
import h5py
import base64
import mrcfile
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import ndimage as ndi
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, callback, clientside_callback

config = None
selected_file = ''
umap = pd.DataFrame(columns=['class', 'x', 'y', 'labels', 'current_class'])
subcluster_umap = pd.DataFrame(columns=['class', 'x', 'y', 'labels', 'current_class'])
tomo = None
instances = None

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE, dbc.icons.FONT_AWESOME])
server = app.server

app.layout = dbc.Container(
    [
        html.Div(["SimSiam embeddings visualization"], className="bg-primary text-white h3 p-2"),
        html.Hr(),
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div(["Upload the config file for running SimSiam:"]),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False),
            html.H6('No config file selected', id='selected_file')])), width=6)),
        html.Hr(),
        dbc.Row(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.H6('Select a file for visualization:'),
                dbc.Spinner(dcc.Dropdown([''], '', id='file-dropdown'), color="primary", type="grow")])
        ])), width=6)),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    [
                        dbc.CardHeader("Clusters UMAP"),
                        dbc.CardBody(
                            [
                                html.Div(["Select a file to visualize"], id='umap-plot-info'),
                                dbc.Col(dbc.Spinner(dcc.Graph(id='umap-plot'), color="primary", type="grow"),
                                        width='auto')
                            ])
                    ]), width=6),
                dbc.Col(dbc.Card(
                    [
                        dbc.CardHeader("Structure view"),
                        dbc.CardBody([
                            html.Div(["Select a point the the UMAP plot"], id='selected-structure-id'),
                            dbc.Col(
                                dbc.Spinner(dcc.Graph(id='selected-structure'), color="primary", type="grow"),
                                width='auto')
                        ])
                    ]), width=6)
            ]),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Subcluster UMAP"),
                    dbc.CardBody([
                        html.Div(["Select a point the the UMAP plot"], id='subcluster-umap-plot-info'),
                        dbc.Col(dbc.Spinner(dcc.Graph(id='subcluster-umap-plot'), color="primary", type="grow"),
                                width='auto')
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Subcluster structure view"),
                    dbc.CardBody([
                        html.Div(["Select a point the the UMAP plot"], id='subcluster-selected-structure-info'),
                        dbc.Col(
                            dbc.Spinner(dcc.Graph(id='subcluster-selected-structure'), color="primary", type="grow"),
                            width='auto')
                    ])
                ]), width=6)
            ])
    ],
    fluid=True,
)


def generate_kmeans_scatter_plot():
    global umap
    global selected_file
    umap['current_class'] = [row['class'] if selected_file in row['labels'] else '0' for i, row in
                             umap.iterrows()]
    selected_samples = umap[umap['current_class'] != '0']
    fig = px.scatter(selected_samples.sort_values('current_class'),
                     x='x', y='y', color='current_class',
                     hover_data=selected_samples.columns, opacity=0.5,
                     color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=600)
    return fig


def generate_subcluster_scatter_plot():
    global subcluster_umap
    global selected_file
    subcluster_umap['current_class'] = [row['class'] if selected_file in row['labels'] else '0' for i, row in
                                        subcluster_umap.iterrows()]
    selected_samples = subcluster_umap[subcluster_umap['current_class'] != '0']
    fig = px.scatter(selected_samples.sort_values('current_class'),
                     x='x', y='y', color='current_class',
                     hover_data=selected_samples.columns, opacity=0.5,
                     color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=600)
    return fig


def load_data_files():
    global tomo
    global instances
    global selected_file
    global config
    tomo = mrcfile.open(os.path.join(config['data_folder'], selected_file)).data
    instances_file = os.path.join(config['instances_mask_folder'],
                                  selected_file.split(config['file_extension'])[0] + '_instance_preds.h5')
    with h5py.File(instances_file, 'r') as f:
        instances = f['instances'][()]


def generate_particle_plot(instance_id):
    global tomo
    global instances
    mask = instances == instance_id
    slices = ndi.find_objects(mask)[0]
    sub_mask = mask[slices]
    patch = tomo[slices].copy()
    patch[sub_mask != 1] = 0
    patch_size = [64, 64, 64]
    pad_size = tuple([(max(math.ceil((p - b) / 2), 0), max(math.floor((p - b) / 2), 0)) for p, b in
                      zip(patch_size, patch.shape)])
    patch = np.pad(patch, pad_size, 'constant')[:patch_size[0], :patch_size[1], :patch_size[2]]
    z, y, x = np.mgrid[:patch.shape[0], :patch.shape[1], :patch.shape[2]]
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=patch.flatten(),
        isomin=0.01,
        isomax=0.99,
        opacity=0.2,
        colorscale='gray'
    ))
    return fig


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        global config
        global selected_file
        global umap
        config = yaml.safe_load(io.StringIO(decoded.decode('utf-8')))
        umap = pd.read_csv(os.path.join(config['prediction_folder'], 'kmeans_clusters_umap_data.csv'))
        umap['class'] = umap['class'].apply(str)
        files = list({'_'.join(x.split('_')[:-1]) for x in umap['labels'].values.tolist()})
        selected_file = files[0]
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return filename, files, selected_file


@callback([
    Output('selected_file', 'children'),
    Output('file-dropdown', 'options'),
    Output('file-dropdown', 'value')
],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True)
def update_upload_output(content, name):
    if content is not None:
        res = parse_contents(content, name)
        return res


@callback(Output('umap-plot', 'figure', allow_duplicate=True),
          Input('file-dropdown', 'value'),
          prevent_initial_call=True)
def update_output(value):
    global selected_file
    selected_file = value
    fig = generate_kmeans_scatter_plot()
    load_data_files()
    return fig


@app.callback([Output('selected-structure', 'figure'),
               Output('subcluster-selected-structure', 'figure',  allow_duplicate=True),
               Output('subcluster-umap-plot', 'figure')],
              Input('umap-plot', 'clickData'),
              prevent_initial_call=True)
def display_click_image(click_data):
    global subcluster_umap
    global umap
    instance_id = int(click_data["points"][0]['customdata'][1].split('_')[-1])
    cluster_id = int(click_data["points"][0]['customdata'][0])
    subcluster_file_path = os.path.join(config['prediction_folder'],
                                        f'subcluster_clusters_selected_{cluster_id}_umap_data.csv')
    if os.path.exists(subcluster_file_path):
        subcluster_umap = pd.read_csv(subcluster_file_path)
    else:
        subcluster_umap = umap[umap['current_class'] == str(cluster_id)]
    fig = generate_subcluster_scatter_plot()
    vol = generate_particle_plot(instance_id)
    return vol, vol, fig


@app.callback(Output('subcluster-selected-structure', 'figure'),
              Input('subcluster-umap-plot', 'clickData'),
              prevent_initial_call=True)
def display_click_image_second_plot(click_data):
    instance_id = int(click_data["points"][0]['customdata'][1].split('_')[-1])
    vol = generate_particle_plot(instance_id)
    return vol


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)
