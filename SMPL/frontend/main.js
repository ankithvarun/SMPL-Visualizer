import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader'
import Stats from 'three/examples/jsm/libs/stats.module'
import { GUI } from 'dat.gui'
import axios from 'axios'

async function getFaces() {
    const response = await axios.get('http://localhost:5000/smpl')
    return response.data.faces
}

async function getVertices(shape_params, pose_params) {
    let payload = {
        shape: shape_params,
        pose: pose_params
    }
    const response = await axios.post('http://localhost:5000/smpl', payload)
    return response.data.vertices
}

const createSMPLGeometry = (verts, faces) => {
    let vertices = new Float32Array(verts)
    let geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
    geometry.setIndex(faces)
    geometry.computeVertexNormals()
    return geometry
}

const refreshGeometry = () => {
    let body_geometry = createSMPLGeometry(verts, faces)
    body.geometry.dispose()
    body.geometry = body_geometry
}

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x050505)

let ambientLight = new THREE.AmbientLight(0x222222);
scene.add(ambientLight);
let light1 = new THREE.DirectionalLight(0xffffff, 0.5);
light1.position.set(1, 1, -1);
scene.add(light1);
let light2 = new THREE.DirectionalLight(0xffffff, 1);
light2.position.set(1, 0, 1);
scene.add(light2);

const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
)
camera.position.set(0, 0, 1.5)

let shape_params = new Array(10).fill(0)
let pose_params = new Array(72).fill(0)
const faces = await getFaces()
let verts = await getVertices(shape_params, pose_params)
const body_material = new THREE.MeshStandardMaterial({color: 0x90ee90, flatShading:true, side:THREE.DoubleSide});
let body_geometry = createSMPLGeometry(verts, faces)
let body = new THREE.Mesh(body_geometry, body_material)
body.position.set(0, 0.3, 0)
scene.add(body)

const renderer = new THREE.WebGLRenderer({antialias: true})
renderer.setSize(window.innerWidth, window.innerHeight)
document.body.appendChild(renderer.domElement)

const controls = new OrbitControls(camera, renderer.domElement)
controls.enableDamping = true
controls.target.set(0, 0, 0)

// let mixer
let modelReady = false

window.addEventListener('resize', onWindowResize, false)
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    render()
}

const stats = new Stats()
document.body.appendChild(stats.dom)

let rotVects = function (x, y, z){
    this.x = x;
    this.y = y;
    this.z = z;
}

let poseDict = {
    'Body': 1,
    'Left Leg': 2,
    'Right Leg': 3,
    'Waist': 4,
    'Left Knee': 5,
    'Right Knee': 6,
    'Torso': 7,
    'Left Foot': 8,
    'Right Foot': 9,
    'Chest': 10,
    'Left Toes': 11,
    'Right Toes': 12,
    'Lower Neck': 13,
    'Left Shoulder': 14,
    'Right Shoulder': 15,
    'Upper Neck': 16,
    'Left Arm': 17,
    'Right Arm': 18,
    'Left Elbow': 19,
    'Right Elbow': 20,
    'Left Wrist': 21,
    'Right Wrist': 22,
    'Left Fingers': 23,
    'Right Fingers': 24,
}

let SMPLParams = function () {
    this.shapeParam1 = 0;
    this.shapeParam2 = 0;
    this.shapeParam3 = 0;
    this.shapeParam4 = 0;
    this.shapeParam5 = 0;
    this.shapeParam6 = 0;
    this.shapeParam7 = 0;
    this.shapeParam8 = 0;
    this.shapeParam9 = 0;
    this.shapeParam10 = 0;
    this.poseParam1 = new rotVects(0, 0, 0);
    this.poseParam2 = new rotVects(0, 0, 0);
    this.poseParam3 = new rotVects(0, 0, 0);
    this.poseParam4 = new rotVects(0, 0, 0);
    this.poseParam5 = new rotVects(0, 0, 0);
    this.poseParam6 = new rotVects(0, 0, 0);
    this.poseParam7 = new rotVects(0, 0, 0);
    this.poseParam8 = new rotVects(0, 0, 0);
    this.poseParam9 = new rotVects(0, 0, 0);
    this.poseParam10 = new rotVects(0, 0, 0);
    this.poseParam11 = new rotVects(0, 0, 0);
    this.poseParam12 = new rotVects(0, 0, 0);
    this.poseParam13 = new rotVects(0, 0, 0);
    this.poseParam14 = new rotVects(0, 0, 0);
    this.poseParam15 = new rotVects(0, 0, 0);
    this.poseParam16 = new rotVects(0, 0, 0);
    this.poseParam17 = new rotVects(0, 0, 0);
    this.poseParam18 = new rotVects(0, 0, 0);
    this.poseParam19 = new rotVects(0, 0, 0);
    this.poseParam20 = new rotVects(0, 0, 0);
    this.poseParam21 = new rotVects(0, 0, 0);
    this.poseParam22 = new rotVects(0, 0, 0);
    this.poseParam23 = new rotVects(0, 0, 0);
    this.poseParam24 = new rotVects(0, 0, 0);
    this.curPoseParamLabel = 'Body';
};

let smplParams = new SMPLParams()

async function updateControllers(value) {
    for (let i = 0 ; i < 3 ; i++){
        console.log(curPoseInd)
        controllers[curPoseInd - 1][i].__li.style.display = "none";
        controllers[poseDict[value] - 1][i].__li.style.display = "";
    }
}

const gui = new GUI()
const shapeParamsFolder = gui.addFolder('Shape Parameters')
shapeParamsFolder.open()
for (let i = 1; i <= 10; i++) {
    shapeParamsFolder.add(smplParams, 'shapeParam' + i, -1, 1).name('Shape PC ' + i).step(0.01).onChange(async function (value) {
        shape_params[i - 1] = value;
        verts = await getVertices(shape_params, pose_params)
    });
}
const poseParamsFolder = gui.addFolder('Pose Parameters')
poseParamsFolder.open()
let controllers = [];
let curPoseInd = 1;
poseParamsFolder.add(smplParams, 'curPoseParamLabel', Object.keys(poseDict)).name('Joint').onChange(async function (value) {
    await updateControllers(value);
    curPoseInd = poseDict[value];
});
for (let i = 1 ; i <= 24; i++){
    let sliders = []
    let x_slider = poseParamsFolder.add(smplParams['poseParam' + i], 'x', -1, 1).name('x').step(0.01).onChange(async function (value) {
        pose_params[3 * (poseDict[smplParams.curPoseParamLabel] - 1)] = value;
        verts = await getVertices(shape_params, pose_params)
    });
    if (i !== 1)
        x_slider.__li.style.display = "none";

    let y_slider = poseParamsFolder.add(smplParams['poseParam' + i], 'y', -1, 1).name('y').step(0.01).onChange(async function (value) {
        pose_params[3 * (poseDict[smplParams.curPoseParamLabel] - 1) + 1] = value;
        verts = await getVertices(shape_params, pose_params)
    });
    if (i !== 1)
        y_slider.__li.style.display = "none";

    let z_slider = poseParamsFolder.add(smplParams['poseParam' + i], 'z', -1, 1).name('z').step(0.01).onChange(async function (value) {
        pose_params[3 * (poseDict[smplParams.curPoseParamLabel] - 1) + 2] = value;
        verts = await getVertices(shape_params, pose_params)
    });
    if (i !== 1)
        z_slider.__li.style.display = "none";

    sliders.push(x_slider);
    sliders.push(y_slider);
    sliders.push(z_slider);
    controllers.push(sliders);
}

const clock = new THREE.Clock()

function animate() {
    requestAnimationFrame(animate)

    controls.update()

    render()

    stats.update()
}

function render() {
    refreshGeometry()
    renderer.render(scene, camera)
}

animate()