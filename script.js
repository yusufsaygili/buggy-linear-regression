let x_vals = [];
let y_vals = [];

let a, m;


const numberofIteration=100
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const {
    canvas,
    ctx
} = setup();



canvasEventListener(canvas, ctx);

function setup() {

    a = tf.variable(tf.scalar(Math.random(1)));
    m = tf.variable(tf.scalar(Math.random(1)));

    const canvas = document.createElement("canvas");
    canvas.id = "canvasLayer";
    canvas.width = screen.width/2;
    canvas.height = screen.height/2;

    const body = document.getElementsByTagName("body")[0];
    body.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    return {
        canvas,
        ctx
    };
}

function canvasEventListener(canvas, ctx) {
    canvas.addEventListener(
        "click",
        function (evt) {
            let mousePos = getMousePos(canvas, evt);
            let x = mapMouseCoordinates(mousePos.x, 0, canvas.width, 0, 1);
            let y = mapMouseCoordinates(mousePos.y, 0, canvas.height, 0, canvas.height / canvas.width);
            x_vals.push(x);
            y_vals.push(y);
            // console.log(mousePos.x + "," + mousePos.y);
            var clicked = true;
            draw(canvas, ctx);
        },
        false
    );
}


async function draw(canvas, ctx) {
    tf.tidy(() => {
        if (x_vals.length > 0) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "rgba(255, 0, 0, 0.2)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            fill = false;
        }

        const ys = tf.tensor1d(y_vals);
        for (let i = 0; i < numberofIteration; i++) {

            optimizer.minimize(() => {
                return loss(predict(x_vals), ys)
            });

        }
        hello()

    });
}
async function hello() {

    for (let i = 0; i < x_vals.length; i++) {
        const px = mapMouseCoordinates(x_vals[i], 0, 1, 0, canvas.width);
        const py = mapMouseCoordinates(y_vals[i], 0, canvas.height / canvas.width, 0, canvas.height);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.fillStyle = "#FFFFFF";
        ctx.fill();
    }
    const lineX = [0, 1];

    const ys = tf.tidy(() => predict(lineX));
    let lineY = ys.dataSync();
    ys.dispose();   
    const x1 = mapMouseCoordinates(lineX[0], 0, 1, 0, canvas.width);
    const x2 = mapMouseCoordinates(lineX[1], 0, canvas.height / canvas.width, 0, canvas.height);
    const y1 = mapMouseCoordinates(lineY[0], 0, 1, 0, canvas.width);
    const y2 = mapMouseCoordinates(lineY[1], 0, canvas.height / canvas.width, 0, canvas.height);

    ctx.beginPath();
    ctx.strokeStyle = 'blue';
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}


function loss(pred, label) {
    const loss = pred.sub(label).square().mean();
    return loss;
}

function predict(x) {
    const xs = tf.tensor1d(x);
    const ys = xs.mul(a).add(m);
    return ys;
}

function getMousePos(canvas, evt) {
    var rect = canvas.getBoundingClientRect();
    return {
        x: Math.round(evt.clientX - rect.left),
        y: Math.round(evt.clientY - rect.top),
    };
}

function mapMouseCoordinates(num, in_min, in_max, out_min, out_max) {
    return (
        ((num - in_min) * (out_max - out_min)) / (in_max - in_min) + out_min
    );
}