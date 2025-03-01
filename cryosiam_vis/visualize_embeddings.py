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
umap = pd.DataFrame(columns=['class', 'x', 'y', 'label'])
tomo = None
instances = None
position = 0
mask = None
current_subtomo = None
current_submask = None
sliding_axis = 'z'
view_type = 'image'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
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
                        dbc.CardHeader("Embeddings UMAP"),
                        dbc.CardBody(
                            [
                                html.Div(dcc.Input(id="selected-instance-id", type="number", debounce=True,
                                                   placeholder="Select a specific instance"), id='umap-plot-info'),
                                dbc.Col(dbc.Spinner(dcc.Graph(id='umap-plot'), color="primary", type="grow"),
                                        width='auto')
                            ])
                    ]), width=6),
                dbc.Col(dbc.Card(
                    [
                        dbc.CardHeader("Selected particle"),
                        dbc.CardBody([
                            html.Div(["Select a point the the UMAP plot"], id='selected-structure-info'),
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
                    dbc.CardHeader("Selected instance embedding UMAP"),
                    dbc.CardBody([
                        html.Div(["Select a point the the UMAP plot"], id='selected-umap-plot-info'),
                        dbc.Col(dbc.Spinner(dcc.Graph(id='selected-umap-plot'), color="primary", type="grow"),
                                width='auto')
                    ])
                ]), width=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Selected particle view"),
                    dbc.CardBody([
                        html.Div(["Axis:"]),
                        dcc.Dropdown(['z', 'y', 'x'], sliding_axis, id='sliding-axis'),
                        html.Div(["View type:"]),
                        dcc.Dropdown(['image', 'mask'], view_type, id='view-type'),
                        dbc.Col([dbc.Spinner(dcc.Graph(id='tomo-slice', style={'width': '500', 'height': '500'}),
                                             color="primary", type="grow")], width='auto')
                    ])
                ]), width=6)

            ])
    ],
    fluid=True,
)


def generate_scatter_plot():
    global config
    global umap
    global selected_file
    fig = px.scatter(umap, x='x', y='y',
                     color='semantic_class2' if 'semantic_class2' in umap.columns else 'semantic_class' if 'semantic_class' in umap.columns else 'log_area',
                     hover_data=umap.columns, opacity=0.5,
                     color_discrete_sequence=px.colors.qualitative.Light24, width=600, height=600)
    return fig


def generate_selected_scatter_plot(selected_instance_id):
    global config
    global umap
    global selected_file
    fig = px.scatter(umap, x='x', y='y',
                     hover_data=umap.columns, opacity=0.5, width=600, height=600)
    selected_point = umap.loc[umap['label'] == selected_instance_id]
    fig.add_trace(go.Scatter(x=selected_point['x'], y=selected_point['y'],
                             mode='markers', marker_line_width=2, marker_size=20, marker_symbol='circle-open-dot'))
    return fig


def load_data_files():
    global tomo
    global instances
    global selected_file
    global config
    global umap
    file_path = os.path.join(config['visualization']['prediction_folder'],
                             f'{selected_file.split(config["file_extension"])[0]}_embeds_umap_data.csv')
    umap = pd.read_csv(file_path)
    if 'semantic_class' in umap.columns:
        umap['semantic_class'] = umap['semantic_class'].apply(str)
    if 'semantic_class2' in umap.columns:
        umap['semantic_class2'] = umap['semantic_class'].apply(str)
    tomo = mrcfile.open(os.path.join(config['data_folder'], selected_file)).data
    instances_file = os.path.join(config['instances_mask_folder'],
                                  selected_file.split(config['file_extension'])[0] + '_instance_preds.h5')
    with h5py.File(instances_file, 'r') as f:
        instances = f['instances'][()]


def generate_particle_plot(instance_id):
    global tomo
    global current_subtomo
    global current_submask
    global instances
    if instance_id != 0:
        mask = instances == instance_id
        slices = ndi.find_objects(mask)[0]
        sub_mask = mask[slices]
        patch = tomo[slices].copy()
        current_subtomo = tomo[slices].copy()
        patch[sub_mask != 1] = 0
        patch_size = [64, 64, 64]
        pad_size = tuple([(max(math.ceil((p - b) / 2), 0), max(math.floor((p - b) / 2), 0)) for p, b in
                          zip(patch_size, patch.shape)])
        patch = np.pad(patch, pad_size, 'constant')[:patch_size[0], :patch_size[1], :patch_size[2]]
        current_subtomo = np.pad(current_subtomo, pad_size, 'constant')[:patch_size[0], :patch_size[1], :patch_size[2]]
        current_submask = np.pad(sub_mask, pad_size, 'constant')[:patch_size[0], :patch_size[1], :patch_size[2]]
    else:
        patch = np.zeros((64, 64, 64))
        current_subtomo = np.zeros((64, 64, 64))
        current_submask = np.zeros((64, 64, 64))
    z, y, x = np.mgrid[:patch.shape[0], :patch.shape[1], :patch.shape[2]]
    fig = go.Figure(data=go.Volume(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        value=patch.flatten(),
        isomin=0.01,
        isomax=0.99,
        opacity=0.3,
        colorscale='gray'
    ))
    return fig


def plot_image():
    global current_subtomo
    global current_submask
    global sliding_axis
    global view_type

    if view_type == 'image':
        fig = px.imshow(current_subtomo, animation_frame=0 if sliding_axis == 'z' else 1 if sliding_axis == 'y' else 2,
                        color_continuous_scale='gray')
    else:
        fig = px.imshow(current_submask, animation_frame=0 if sliding_axis == 'z' else 1 if sliding_axis == 'y' else 2,
                        color_continuous_scale='gray')
    return fig


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        global config
        global selected_file
        config = yaml.safe_load(io.StringIO(decoded.decode('utf-8')))
        files = [x.split('_embeds_umap_data.csv')[0] + config['file_extension'] for x in
                 os.listdir(config['visualization']['prediction_folder']) if x.endswith('_embeds_umap_data.csv')]
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
    load_data_files()
    fig = generate_scatter_plot()
    return fig


@app.callback([Output('selected-structure', 'figure'),
               Output('selected-structure-info', 'children'),
               Output('tomo-slice', 'figure'),
               Output('selected-umap-plot', 'figure'),
               Output('selected-umap-plot-info', 'children')],
              Input('umap-plot', 'clickData'),
              prevent_initial_call=True)
def display_click_image(click_data):
    instance_id = int(click_data["points"][0]['customdata'][1])
    class_id = int(click_data["points"][0]['customdata'][-2])
    vol = generate_particle_plot(instance_id)
    message = f"Class id: {class_id}, Instance id: {instance_id}"
    fig = plot_image()
    selected_fig = generate_selected_scatter_plot(instance_id)
    return vol, message, fig, selected_fig, message


@app.callback([Output('selected-structure', 'figure', allow_duplicate=True),
               Output('selected-structure-info', 'children', allow_duplicate=True),
               Output('tomo-slice', 'figure', allow_duplicate=True)],
              Input('selected-umap-plot', 'clickData'),
              prevent_initial_call=True)
def display_click_second_image(click_data):
    instance_id = int(click_data["points"][0]['customdata'][1])
    class_id = int(click_data["points"][0]['customdata'][-2])
    vol = generate_particle_plot(instance_id)
    message = f"Class id: {class_id}, Instance id: {instance_id}"
    fig = plot_image()
    return vol, message, fig


@callback(
    [Output('selected-structure', 'figure', allow_duplicate=True),
     Output('selected-structure-info', 'children', allow_duplicate=True),
     Output('tomo-slice', 'figure', allow_duplicate=True),
     Output('selected-umap-plot', 'figure', allow_duplicate=True),
     Output('selected-umap-plot-info', 'children', allow_duplicate=True)],
    Input("selected-instance-id", "value"),
    prevent_initial_call=True
)
def number_render(val):
    global umap
    vol = generate_particle_plot(val)
    row = umap[umap['label'] == val]
    class_id = row['semantic_class_2'] if 'semantic_class_2' in umap.columns else row['semantic_class']
    message = f"Class id: {class_id.values[0]}, Instance id: {val}"
    fig = plot_image()
    selected_fig = generate_selected_scatter_plot(val)
    return vol, message, fig, selected_fig, message


@callback(
    Output('tomo-slice', 'figure', allow_duplicate=True),
    Input('sliding-axis', 'value'),
    prevent_initial_call=True
)
def update_axis(value):
    global sliding_axis
    sliding_axis = value
    fig = plot_image()
    return fig


@callback(
    Output('tomo-slice', 'figure', allow_duplicate=True),
    Input('view-type', 'value'),
    prevent_initial_call=True
)
def update_axis(value):
    global view_type
    view_type = value
    fig = plot_image()
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, dev_tools_props_check=False)
