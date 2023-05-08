from flask import Flask, request
from flask_cors import CORS
from smpl import SMPLModel
import torch
import numpy as np
import os

def create_geometry(model, shape_params, pose_params):
    betas = torch.from_numpy(shape_params).type(torch.float64).to(device)
    pose = torch.from_numpy(pose_params).type(torch.float64).to(device)
    trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)

    verts = model(betas, pose, trans).cpu().detach().numpy().ravel()
    faces = model.faces.ravel()

    return verts, faces

pose_size = 72
beta_size = 10

gpu_id = [1]
if len(gpu_id) > 0 and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print(device)

model = SMPLModel(device=device)
verts, faces = create_geometry(model, np.zeros(beta_size), np.zeros(pose_size))

app = Flask(__name__)
CORS(app)

@app.route('/smpl', methods=['GET', 'POST'])
def get_geometry():
    if request.method == "GET":
        return {"faces": faces.tolist()}
    elif request.method == "POST":
        request_data = request.get_json()
        shape_params = np.asarray(request_data["shape"])
        pose_params = np.asarray(request_data["pose"])
        
        verts, _ = create_geometry(model, shape_params, pose_params)
        return {"vertices": verts.tolist()}
