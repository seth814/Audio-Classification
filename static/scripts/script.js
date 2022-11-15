//set up variables

const body = document.querySelector('body');
const menu = document.querySelector('.menu-container');
const button = document.querySelector('.center-button');
const buttonText = document.querySelector('.button-text');
const canvas = document.querySelector('.visualizer');
const staticLine = document.querySelector('.static-line');
const visContainer = document.querySelector('.vis-container');

//create web audio api context

let audioCtx;
const canvasCtx = canvas.getContext("2d");

//variable to track the state of center button
let startState = true;
visContainer.hidden = true;

//main block for doing the audio recording

if (navigator.mediaDevices.getUserMedia) {
    console.log('getUserMedia supported.')
    const constraints = { audio: true };
    let chunks = [];

    let onSuccess = function(stream) {
        const mediaRecorder = new MediaRecorder(stream);

        visualize(stream)
        
        button.onclick = function() {
            if (startState) {
                mediaRecorder.start();
                console.log(mediaRecorder.state);
                console.log("recorder started");
                body.style.backgroundColor = "#307D92";
                menu.style.backgroundColor = "#307D92";
                button.style.backgroundColor = "#8BE451"    
                buttonText.textContent = "STOP";
                startState = false;
                visContainer.hidden = false;
                staticLine.hidden = true;
            } else {
                mediaRecorder.stop();
                console.log(mediaRecorder.state);
                console.log("recorder stopped");
                body.style.backgroundColor = "#8BE451"; //AEE13F
                menu.style.backgroundColor = "#8BE451";
                button.style.backgroundColor = "#318BB1";    
                buttonText.textContent = "START";
                startState = true;
                visContainer.hidden = true;
                staticLine.hidden = false;
            }
        }
        
        mediaRecorder.onstop = function(e) {
            console.log("data available after MediaRecorder.stop() called.");
            
            const blob = new Blob(chunks, { 'type' : 'audio/wav' });
            chunks = [];
            const audioUrl = URL.createObjectURL(blob);
            const audio = new Audio(audioUrl);
            audio.play();
            console.log("recorder stopped");

            let data = new FormData()
            data.append('file', blob , 'file')

            fetch('http://127.0.0.1:8080/receive', {
                method: 'POST',
                body: data

            }).then(response => response.json()
            ).then(response => {
                let para = document.createElement("p");
                let node = document.createTextNode(response);
                para.appendChild(node);
                para.setAttribute(
                    'style',
                    'font-size: 60px;color: #FFFFFF;justify-self:center',
                );

                let middle = document.getElementById('middleid');
                while (middle.firstChild) {
                    middle.removeChild(middle.firstChild);
                }
                middle.appendChild(para);
                
                console.log(response)
            });
            

        }

        mediaRecorder.ondataavailable = function(e) {
            chunks.push(e.data);
        }
    }

    let onError = function(err) {
        console.log('The following error occured: ' + err);
    }

    navigator.mediaDevices.getUserMedia(constraints).then(onSuccess, onError);

} else {
    console.log('getUserMedia not supported on your browser!');
}


function visualize(stream) {
    if(!audioCtx) {
        audioCtx = new AudioContext();
    }

    const source = audioCtx.createMediaStreamSource(stream);

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    source.connect(analyser);
    //analyser.connect(audioCtx.destination);

    draw()

    function draw() {
        const WIDTH = canvas.width
        const HEIGHT = canvas.height;

        requestAnimationFrame(draw);

        analyser.getByteTimeDomainData(dataArray);

        if (startState) {
            canvasCtx.fillStyle = "#8BE451";
        } else {
            canvasCtx.fillStyle = "#307D92";
        }
        canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

        canvasCtx.lineWidth = 3;
        canvasCtx.strokeStyle = '#FFFFFF';

        canvasCtx.beginPath();

        let sliceWidth = WIDTH * 1.0 / bufferLength;
        let x = 0;


        for(let i = 0; i < bufferLength; i++) {

            let v = dataArray[i] / 128.0;
            let y = v * HEIGHT/2;

            if(i === 0) {
                canvasCtx.moveTo(x, y);
            } else {
                canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
        }

        canvasCtx.lineTo(canvas.width, canvas.height/2);
        canvasCtx.stroke();

    }

    window.onresize = function() {
        canvas.width = window.innerWidth;
    }
    window.onresize();
}

