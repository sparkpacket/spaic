// Orchestrates UI and connects TFModel + NeuroAgents.
// Also contains a tiny tokenizer.

class Tokenizer {
  constructor() {
    this.chars = new Set();
    this.charList = [];
    this.charToIdx = {};
    this.idxToChar = {};
    this.vocabSize = 0;
  }
  buildFromText(text) {
    this.chars = new Set(text.split(''));
    this.charList = Array.from(this.chars).sort();
    // ensure stable small vocab by adding space and common punctuation
    const essentials = [' ','\n','.','!','?',',','-','"',"'",':',';'];
    essentials.forEach(c=>{ if(!this.chars.has(c)){ this.charList.push(c); }});
    // map
    this.charList = Array.from(new Set(this.charList)); // unique
    this.charToIdx = {}; this.idxToChar = {};
    this.charList.forEach((c,i)=>{ this.charToIdx[c]=i; this.idxToChar[i]=c; });
    this.vocabSize = this.charList.length || 1;
  }
  textToIndices(text) {
    if (this.vocabSize===0) this.buildFromText(text);
    const arr = [];
    for (let i=0;i<text.length;i++){
      const c = text[i];
      const idx = (this.charToIdx.hasOwnProperty(c)) ? this.charToIdx[c] : 0;
      arr.push(idx);
    }
    return arr;
  }
  indexToChar(i){ return this.idxToChar[i] || ' '; }
}

const state = {
  tokenizer: new Tokenizer(),
  tfModel: null,
  neuroAgents: [],
  agentCounter: 0,
  selfLoop: null
};

function ui(id) { return document.getElementById(id); }

async function refreshTokenizer() {
  const text = ui('corpus').value;
  state.tokenizer.buildFromText(text || " ");
  return state.tokenizer;
}

async function trainTfShort() {
  await refreshTokenizer();
  if (!state.tfModel) state.tfModel = new TFModel(state.tokenizer, parseInt(ui('seqLen').value||20), 48);
  ui('trainTfBtn').disabled = true;
  await state.tfModel.trainOnText(ui('corpus').value, 4, 32);
  ui('trainTfBtn').disabled = false;
  alert('TF model trained briefly (tiny).');
}

function spawnNeuroAgent() {
  const id = `A${++state.agentCounter}`;
  const seqLen = parseInt(ui('seqLen').value||20);
  const agent = new NeuroAgent(id, seqLen, 20);
  state.neuroAgents.push(agent);
  renderAgentList();
  return agent;
}

async function evolvePopulation() {
  const pop = parseInt(ui('popSize').value||10);
  const seqLen = parseInt(ui('seqLen').value||20);
  await refreshTokenizer();
  const ds = makeNeatDatasetFromText(ui('corpus').value, state.tokenizer, seqLen, 800);
  if (!ds || ds.length===0) { alert('Not enough corpus for evolution'); return; }
  // make initial population
  state.neuroAgents = [];
  for (let i=0;i<pop;i++) {
    const id = `G${i+1}`;
    const agent = new NeuroAgent(id, seqLen, 8 + Math.floor(Math.random()*28));
    state.neuroAgents.push(agent);
  }
  renderAgentList();
  // evolve each agent a little (quick rounds)
  ui('evolveBtn').disabled = true;
  for (let i=0;i<state.neuroAgents.length;i++) {
    const a = state.neuroAgents[i];
    // evolve on small iterations to keep browser responsive
    await a.evolveOnDataset(ds, {iterations: 40});
    renderAgentList();
    await new Promise(r=>setTimeout(r, 80));
  }
  ui('evolveBtn').disabled = false;
  alert('Evolved population (quick).');
}

async function stepSelfTrain() {
  // Each agent (TF + neuro) generates text; that text is appended to corpus (self-learning)
  await refreshTokenizer();
  // TF generation
  if (state.tfModel) {
    const gen = await state.tfModel.generate("S:", 150, 0.9);
    ui('corpus').value += "\n" + gen;
  }
  // Neuro agents: each produces short sequence by iteratively predicting normalized char index
  for (let agent of state.neuroAgents) {
    const seqLen = agent.seqLen;
    // seed from random slice of current corpus
    const corp = ui('corpus').value;
    const start = Math.max(0, Math.floor(Math.random()*(Math.max(1, corp.length - seqLen))));
    const seed = corp.slice(start, start+seqLen);
    const toks = state.tokenizer.textToIndices(seed);
    const norm = toks.map(x=>x/(state.tokenizer.vocabSize-1));
    let out = "";
    let cur = norm.slice();
    for (let i=0;i<80;i++){
      const val = agent.activate(cur);
      const idx = Math.round(val*(state.tokenizer.vocabSize-1));
      out += state.tokenizer.indexToChar(idx);
      cur.shift();
      cur.push(idx/(state.tokenizer.vocabSize-1));
    }
    ui('corpus').value += "\n" + out;
  }
  // quick small retrain on TF model (tiny)
  if (state.tfModel) {
    await state.tfModel.trainOnText(ui('corpus').value, 2, 32);
  }
  renderAgentList();
}

function renderAgentList() {
  const container = ui('agentList');
  container.innerHTML = '';
  state.neuroAgents.forEach(a=>{
    const el = document.createElement('div'); el.className='agent-card';
    el.innerHTML = `<strong>${a.id}</strong> (seqLen ${a.seqLen}) <br/> nodes: ${a.net.nodes.length} conns: ${a.net.connections.length}`;
    const spawnBtn = document.createElement('button');
    spawnBtn.textContent = 'Spawn child';
    spawnBtn.onclick = ()=> {
      const child = a.spawnChild(`C${++state.agentCounter}`);
      state.neuroAgents.push(child);
      renderAgentList();
    };
    const vizBtn = document.createElement('button');
    vizBtn.textContent = 'Viz';
    vizBtn.onclick = ()=> {
      const canvas = ui('viz');
      const json = a.toJSON();
      drawNetworkOnCanvas(canvas, json);
    };
    el.appendChild(document.createElement('br'));
    el.appendChild(spawnBtn);
    el.appendChild(vizBtn);
    container.appendChild(el);
  });
  ui('agentsInfo').textContent = `Agents: ${state.neuroAgents.length}`;
}

function downloadCorpus() {
  const blob = new Blob([ui('corpus').value], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href=url; a.download='spaic_corpus.txt'; a.click();
  URL.revokeObjectURL(url);
}

function clearCorpus() { ui('corpus').value = ""; }

async function generateTF() {
  if (!state.tfModel) { alert('No TF model yet. Train one (Train TF model)'); return; }
  const out = await state.tfModel.generate("Seed ", 400, 0.8);
  alert('TF generated (first 800 chars):\n\n' + out.slice(0,1200));
}

async function generateNeuro() {
  if (state.neuroAgents.length===0){ alert('No neuro agents. Spawn one.'); return; }
  const a = state.neuroAgents[Math.floor(Math.random()*state.neuroAgents.length)];
  // make a small generation from a
  const seqLen = a.seqLen;
  const corp = ui('corpus').value;
  const start = Math.max(0, Math.floor(Math.random()*(Math.max(1, corp.length - seqLen))));
  const seed = corp.slice(start, start+seqLen);
  const toks = state.tokenizer.textToIndices(seed);
  const norm = toks.map(x=>x/(state.tokenizer.vocabSize-1));
  let out = "";
  let cur = norm.slice();
  for (let i=0;i<200;i++){
    const val = a.activate(cur);
    const idx = Math.round(val*(state.tokenizer.vocabSize-1));
    out += state.tokenizer.indexToChar(idx);
    cur.shift(); cur.push(idx/(state.tokenizer.vocabSize-1));
  }
  alert(`Neuro agent ${a.id} generated:\n\n` + out.slice(0,1200));
}

function wireUi() {
  ui('trainTfBtn').onclick = trainTfShort;
  ui('spawnNeuroBtn').onclick = ()=>{ spawnNeuroAgent(); renderAgentList(); };
  ui('evolveBtn').onclick = evolvePopulation;
  ui('genTfBtn').onclick = generateTF;
  ui('genNeuroBtn').onclick = generateNeuro;
  ui('downloadCorpus').onclick = downloadCorpus;
  ui('clearCorpus').onclick = clearCorpus;
  ui('stepSelfBtn').onclick = stepSelfTrain;
  ui('startLoopBtn').onclick = ()=> {
    ui('startLoopBtn').disabled = true; ui('stopLoopBtn').disabled=false;
    state.selfLoop = setInterval(()=>{ stepSelfTrain(); }, 4000);
  };
  ui('stopLoopBtn').onclick = ()=> {
    clearInterval(state.selfLoop); state.selfLoop = null;
    ui('startLoopBtn').disabled = false; ui('stopLoopBtn').disabled=true;
  };
}

window.addEventListener('load', ()=>{
  wireUi();
  refreshTokenizer().then(()=>{ /* ready */ });
});
