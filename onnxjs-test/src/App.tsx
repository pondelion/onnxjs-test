import React from 'react';
import './App.css';

import { Tensor, InferenceSession } from 'onnxjs';


var sessions: any = {
  "cpu": null,
  "webgl": null,
  "wasm": null,
};

const inference = async (backend: string) => {
  console.log(backend)
  console.log(sessions[backend]);
  if (sessions[backend] === null) {
    // backendHint :cpu / webgl /wasm
    console.log(`initialize ${backend} session : ` + (new Date()).toISOString());
    sessions[backend] = new InferenceSession({ backendHint: backend });
    console.log("initialize done : " + (new Date()).toISOString());
    console.log("load start : " + (new Date()).toISOString());
    await sessions[backend].loadModel('./model.onnx');
    console.log("load done : " + (new Date()).toISOString());
  }
  const dummyInput = new Float32Array(1 * 3 * 224 * 224).fill(0);
  dummyInput[6] = Math.random();
  const inputTensor = new Tensor(dummyInput, 'float32', [1, 3, 224, 224]);

  console.log(`[${backend}]start inference : ` + (new Date()).toISOString());
  const output = await sessions[backend].run([inputTensor])
  console.log(`[${backend}]done inference : ` + (new Date()).toISOString());

  console.log(output);
}

function App() {
  const [backend, setBackend] = React.useState<string>('webgl');

  return (
    <div className="App">
      <div>
        backend : 
        <select value={backend} onChange={(e: any) => {setBackend(e.target.value)}}>
          <option value="webgl">webgl</option>
          <option value="cpu">cpu</option>
          <option value="wasm">wasm</option>
        </select>
      </div>
      <div>
        <button onClick={() => inference(backend)}>
          inference
        </button>
      </div>
    </div>
  );
}

export default App;
