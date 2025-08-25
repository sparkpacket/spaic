// Wraps neataptic to create/evolve lightweight agents for next-char prediction.
// Approach: inputs are the last seqLen character indices scaled to [0,1] (so input size = seqLen).
// output is a single number in [0,1] representing next char index scaled.
// This is a compact representation that keeps networks small for the browser.

class NeuroAgent {
  constructor(id, seqLen=20, hidden=20) {
    this.id = id;
    this.seqLen = seqLen;
    this.net = new neataptic.architect.Perceptron(seqLen, hidden, 1);
    this.type = 'neuro';
  }

  // train via evolve on the provided dataset (array of {input:[...], output:[...]})
  async evolveOnDataset(dataset, options={iterations:100, popsize:30}) {
    // dataset expected as neataptic style training set: [{input:[..], output:[..]},...]
    try {
      // neataptic Network.evolve is async
      await this.net.evolve(dataset, {iterations: options.iterations || 50, error: options.error || 0.1, log: options.log || 0});
      return true;
    } catch (e) {
      console.warn('evolve failed', e);
      return false;
    }
  }

  // activate with a normalized input (array length seqLen)
  activate(normInput) {
    // returns float in [0,1]
    return this.net.activate(normInput)[0];
  }

  // mutate and return a new child agent
  spawnChild(newId) {
    const child = new NeuroAgent(newId, this.seqLen, 0);
    child.net = this.net.clone();
    // apply a few mutations
    child.net.mutate(neataptic.methods.mutation.ADD_NODE);
    child.net.mutate(neataptic.methods.mutation.MOD_WEIGHT);
    child.net.mutate(neataptic.methods.mutation.ADD_CONN);
    return child;
  }

  // quick JSON snapshot for viz
  toJSON() { return this.net.toJSON(); }
}

// helper: make training set from corpus (string)
// returns array of {input:[normIndices], output:[normNextIndex]}
function makeNeatDatasetFromText(text, tokenizer, seqLen=20, maxExamples=2000) {
  const tokens = tokenizer.textToIndices(text);
  const ds = [];
  for (let i=0;i+seqLen<tokens.length && ds.length<maxExamples;i++) {
    const seq = tokens.slice(i, i+seqLen);
    const next = tokens[i+seqLen];
    const normSeq = seq.map(x=> x/(tokenizer.vocabSize-1));
    const normNext = [next/(tokenizer.vocabSize-1)];
    ds.push({input: normSeq, output: normNext});
  }
  return ds;
}
