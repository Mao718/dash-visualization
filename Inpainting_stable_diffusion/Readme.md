# Inpainting Interface
![interface show](inpainting_interface.git)  
This interface using [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) pretrained weights. The user can upload their image to the server, choose the region to be inpainted and give you back the predict image.

Set it up uncomment one of these two to choose you want to host it locally or host it as a server 
```
# set it up locally
#app.run_server(debug=True ,port = 8001 )
# set it up on the server
#app.run_server(debug=True, host='192.168.0.16',port = 8001 )
```
Then run

```python inpaint_dash.py```
