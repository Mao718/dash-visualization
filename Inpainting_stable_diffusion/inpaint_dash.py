import datetime

from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
#import dash_bootstrap_components as dbc
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
import time
from PIL import Image
import plotly.express as px

import cv2 as cv
import io
import base64
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import dash_bootstrap_components as dbc
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to(device)

running = False
app = Dash(__name__)#,external_stylesheets=[dbc.themes.BOOTSTRAP]
app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
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
        multiple=True
    ),
    html.Div(id='output-image-upload'),W
    html.Label('Mask Input'),
    html.Div(["start X" ,dcc.Input(id='x11',value='400', type='text'),"start Y", dcc.Input(id='y11',value='120', type='text'),"End X" ,dcc.Input(id='x12',value="610", type='text'),"End Y", dcc.Input(id='y12',value="650", type='text')]),
    html.Div(["start X" ,dcc.Input(id='x21',value='', type='text'),"start Y", dcc.Input(id='y21',value='', type='text'),"End X" ,dcc.Input(id='x22',value='', type='text'),"End Y", dcc.Input(id='y22',value='', type='text')]),
    html.Div(["start X" ,dcc.Input(id='x31',value='', type='text'),"start Y", dcc.Input(id='y31',value='', type='text'),"End X" ,dcc.Input(id='x32',value='', type='text'),"End Y", dcc.Input(id='y32',value='', type='text')]),
    html.Div(["start X" ,dcc.Input(id='x41',value='', type='text'),"start Y", dcc.Input(id='y41',value='', type='text'),"End X" ,dcc.Input(id='x42',value='', type='text'),"End Y", dcc.Input(id='y42',value='', type='text'),html.Button(id='submit-button-state', n_clicks=0, children='Submit')]),
    dcc.Graph(id='mask'),
    html.Div(["Prompt" ,dcc.Input(id='prompt: ',value='', type='text',style={'width': '80%'}),html.Button(id='submit_generate', n_clicks=0, children='Submit')]),
    dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
    dcc.Graph(id='Result'),
    
])





def parse_contents(contents, filename, date):
    #print(type(contents))
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        # html.Img(src=contents),
        dcc.Graph(id='life-exp-vs-gdp', figure=update_fig(contents)),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
def update_fig(contents):
    #img = Image.open(filename)

    #base64_decoded = base64.b64decode()

    image = decode_img(contents)
    fig = px.imshow(image)
    return fig

def decode_img(contents):
    img_uri = contents
    encoded_img = img_uri.split(",")[1]
    binary = base64.b64decode(encoded_img)
    with open("test_save.png", 'wb') as f:
        f.write(binary)
    image = Image.open("test_save.png")
    image = np.array(image)

    #image = Image.open(io.BytesIO(base64_decoded))
    #image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image

@app.callback( Output('Result', 'figure'),
            Output("loading-output-1", "children"),
            Input('submit_generate', 'n_clicks'),
            State('prompt: ', 'value'),
            State('x11', 'value'),
            State('y11', 'value'),
            State('x12', 'value'),
            State('y12', 'value'),
            State('x21', 'value'),
            State('y21', 'value'),
            State('x22', 'value'),
            State('y22', 'value'),
            State('x31', 'value'),
            State('y31', 'value'),
            State('x32', 'value'),
            State('y32', 'value'),
            State('x41', 'value'),
            State('y41', 'value'),
            State('x42', 'value'),
            State('y42', 'value')
              )
def result( n_clicks , prompt , x11,y11,x12,y12,x21,y21,x22,y22,x31,y31,x32,y32,x41,y41,x42,y42):
    
    if n_clicks == 0:
        return px.imshow(np.zeros((100,100))) ,0
    image = Image.open("test_save.png")
    mask = np.zeros(np.array(image).shape[:2], dtype=np.uint8)
    print("OK")
    if y11 and y12 and x11 and x12:
        mask[int(y11):int(y12),int(x11):int(x12)] = 255
    if y21 and y22 and x21 and x22:
        mask[int(y21):int(y22),int(x21):int(x22)] = 255
    if y31 and y32 and x31 and x32:
        mask[int(y31):int(y32),int(x31):int(x32)] = 255
    if y41 and y42 and x41 and x42:
        mask[int(y41):int(y42),int(x41):int(x42)] = 255
    mask = Image.fromarray(mask)
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    image_o = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    print("img out")
    running = False
    return px.imshow(image_o.resize((np.array(image).shape[1],np.array(image).shape[0]) )) ,0
    


## load the upload image
@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]


        return children

@app.callback(
    Output('mask', 'figure'),
    Input('submit-button-state', 'n_clicks'),
    State('x11', 'value'),
    State('y11', 'value'),
    State('x12', 'value'),
    State('y12', 'value'),
    State('x21', 'value'),
    State('y21', 'value'),
    State('x22', 'value'),
    State('y22', 'value'),
    State('x31', 'value'),
    State('y31', 'value'),
    State('x32', 'value'),
    State('y32', 'value'),
    State('x41', 'value'),
    State('y41', 'value'),
    State('x42', 'value'),
    State('y42', 'value'),
)
def update_mask(n_clicks, x11,y11,x12,y12,x21,y21,x22,y22,x31,y31,x32,y32,x41,y41,x42,y42):
    image = Image.open("test_save.png")
    image = np.array(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if n_clicks == 0:
        return px.imshow(mask) 
    print(mask.shape)
    if y11 and y12 and x11 and x12:
        mask[int(y11):int(y12),int(x11):int(x12)] = 255
    if y21 and y22 and x21 and x22:
        mask[int(y21):int(y22),int(x21):int(x22)] = 255
    if y31 and y32 and x31 and x32:
        mask[int(y31):int(y32),int(x31):int(x32)] = 255
    if y41 and y42 and x41 and x42:
        mask[int(y41):int(y42),int(x41):int(x42)] = 255
    fig = px.imshow(mask, color_continuous_scale='gray')    
    return fig



if __name__ == '__main__':
    # set it up locally
    app.run_server(debug=True ,port = 8001 )
    # set it up on the server
    #app.run_server(debug=True, host='192.168.0.16',port = 8001 )