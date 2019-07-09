var bx;
var by;
var f1;
var f2;
var boxSize = 20;
var overBox = false;
var locked = false;
var xOffset = 0.0; 
var yOffset = 0.0;

// audio
var osc, envelope;
// timing
var thesi;
var bpm;
var tempo; // how long a sixteenth note is in milliseconds
var clock; // the timer for moving from note to note
var bps;
var playStat = true;

// server
var thesiSend = 0;
var responseReceived = true;

// music
var column = [];
var initSound = "acoustic_grand_piano";

// client
var socket;

function preload() {
  ctx = getAudioContext(); // on récupère le contexte audio pour le passer à notre instrument
  lead = Soundfont.instrument(ctx, initSound); // on intialise notre synthétiseur avec le contexte et le nom du son à charger
}

function setup() {
  // put setup code here
  createCanvas(640, 480);
  bx = width/2.0;
  by = height/2.0;
  f1 = map(bx, 0.0, width, 0.3, 0.8);
  f2 = map(by, 0.0, height, 0.1, 0.9);
  rectMode(RADIUS);

  // // sound
  // osc = new p5.SinOsc();
  // envelope = new p5.Env();
  // envelope.setADSR(0.001, 0.5, 0.1, 0.5);
  // envelope.setRange(1, 0);
  // osc.start();
  // timing
  thesi = 0;
  bpm = 120;
  bps = bpm/60.0;
  tempo = int(1000 / (bps * 4));
  clock = millis();
  
  thesiSend = 0;

  // client
  // socket = io.connect('http://localhost:8888');
  namespace = '/test';

  // Connect to the Socket.IO server.
  // The connection URL has the following format:
  //     http[s]://<domain>:<port>[/<namespace>]
  socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);
  // socket = io.connect('http://localhost:8888/test');
  socket.on('send column', function(msg){
    column = msg['column'];
    // console.log(msg['column']);
    responseReceived = true;
  });
  // socket.emit('my event', {'data': 'lala'})
}

function draw() {
  // put drawing code here
  background(255);
  stroke(0);
  fill(200);
  rect(width/2.0,height/2.0, width/2.0-1,height/2.0-1)
  fill(120);
  // Test if the cursor is over the box 
  if (mouseX > bx-boxSize && mouseX < bx+boxSize && 
    mouseY > by-boxSize && mouseY < by+boxSize) {
    overBox = true;  
    if(!locked) {
      stroke(70); 
      fill(70);
    } 
  } else {
    stroke(70);
    fill(70);
    overBox = false;
  }
  // Draw the box
  rect(bx, by, boxSize, boxSize);
  // draw the text
  stroke(0,120);
  fill(0,120);
  text("density=" + nf(f1, 0, 2), 10, 20);
  text("register=" + nf(f2, 0, 2), 10, 40);
  // music
  if ((millis() - clock >= tempo) && (playStat)) {
    // console.log('play ----------');
    for (var i=0; i<column.length; i++){
      if (column[i] > 0){
        play(i-12);
      }
    }
    // send new request to server
    if (thesiSend++ >= 0){
      sendFeatures();
      thesiSend = 0;
    }
    clock = millis();
    sendFeatures();
    // thesiSend++;
    // if (thesiSend > 4) {
    //   thesiSend = 0;
    //   sendFeatures();
    // }
  }
}

function play(midinote) {
  lead.then(function (inst) {
      inst.play(midinote + 24, 0, {
          loop: false
      });
  });
}

function mousePressed() {
  if(overBox) { 
    locked = true; 
    fill(90);
  } else {
    locked = false;
  }
  xOffset = mouseX-bx; 
  yOffset = mouseY-by; 
}

function mouseDragged() {
  if(locked) {
    bx = mouseX-xOffset; 
    by = mouseY-yOffset; 
  }
  if (bx < 0.0) bx = 0.0;
  if (by < 0.0) by = 0.0;
  if (bx > width) bx = width;
  if (by > height) by = height;
  // f1 = bx/width;
  // f2 = by/height;
  f1 = map(bx, 0.0, width, 0.3, 0.8);
  f2 = map(by, 0.0, height, 0.1, 0.9);
}

function mouseReleased() {
  locked = false;
}

function sendFeatures(){
  var allFeatures = [0.0, 0.0];
  var tmpStr = "_";
  for (var i=0; i<allFeatures.length; i++){
    tmpStr += String(allFeatures[i]) + '_';
  }
  // console.log(" --- NEW STUFF --- responseReceived: ", responseReceived);
  // send to server
  if (responseReceived){
    // socket.emit('my event',tmpStr);
    socket.emit('send features', {'f1': f1, 'f2': f2});
    responseReceived = false;
  }
  // start getting response
  // initialise response
  // entireResponse = "";
}

function clientEvent(){
  responseReceived = true;
  console.log("clientEvent");
}