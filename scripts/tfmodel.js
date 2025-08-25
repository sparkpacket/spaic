// Tiny TF model for char-level next-character prediction.
// Uses a small LSTM layer (if available) or dense fallback.
// Exposes: TFModel(tokenizer, seqLen)
class TFModel {
  constructor(tokenizer, seqLen = 20, lstmUnits = 64) {
    this.tokenizer = tokenizer;
    this.seqLen = seqLen;
    this.lstmUnits = lstmUnits;
    this.model = null;
    this.vocabSize = tokenizer.vocabSize;
  }

  async build() {
    // simple sequential LSTM -> dense
    const tfm = tf.sequential();
    // Input shape: [seqLen, features], features = vocabSize (we feed one-hot vectors)
    try {
      tfm.add(tf.layers.lstm({
        units: this.lstmUnits,
        inputShape: [this.seqLen, this.vocabSize],
      }));
    } catch (e) {
      // fallback to flatten -> dense if LSTM missing
      tfm.add(tf.layers.flatten({inputShape: [this.seqLen, this.vocabSize]}));
      tfm.add(tf.layers.dense({units: this.lstmUnits, activation: 'relu'}));
    }
    tfm.add(tf.layers.dense({units: this.vocabSize, activation: 'softmax'}));
    tfm.compile({optimizer: tf.train.adam(0.01), loss: 'categoricalCrossentropy'});
    this.model = tfm;
  }

  // prepare dataset: sequences of seqLen -> next char one-hot
  makeDatasetFromText(text, maxExamples = 2000) {
    const seqLen = this.seqLen;
    const tokens = this.tokenizer.textToIndices(text);
    const X = [], Y = [];
    for (let i = 0; i + seqLen < tokens.length && X.length < maxExamples; i++) {
      const seq = tokens.slice(i, i + seqLen);
      const next = tokens[i + seqLen];
      // one-hot encode
      const xOne = seq.map(idx => {
        const arr = new Float32Array(this.vocabSize);
        arr[idx] = 1;
        return arr;
      });
      X.push(xOne);
      const yOne = new Float32Array(this.vocabSize);
      yOne[next] = 1;
      Y.push(yOne);
    }
    if (X.length === 0) return null;
    const xs = tf.tensor(X); // shape [N, seqLen, vocabSize]
    const ys = tf.tensor(Y); // shape [N, vocabSize]
    return {xs, ys};
  }

  async trainOnText(text, epochs = 6, batchSize = 32) {
    if (!this.model) await this.build();
    const ds = this.makeDatasetFromText(text);
    if (!ds) return {trained:0};
    const {xs, ys} = ds;
    await this.model.fit(xs, ys, {epochs, batchSize, callbacks: {onEpochEnd: async (e,logs) => {
      await tf.nextFrame();
    }}});
    xs.dispose(); ys.dispose();
    return {trained:1};
  }

  // generate text using greedy sampling with optional temperature
  async generate(seed = "Hello ", length = 200, temperature = 1.0) {
    if (!this.model) return "";
    let out = seed;
    let seq = this.tokenizer.textToIndices(seed);
    while (seq.length < this.seqLen) seq.unshift(0); // pad left
    for (let i=0;i<length;i++) {
      const inputSeq = seq.slice(-this.seqLen);
      const xOne = inputSeq.map(idx => {
        const arr = new Float32Array(this.vocabSize);
        arr[idx] = 1;
        return arr;
      });
      const inputTensor = tf.tensor([xOne]); // [1, seqLen, vocabSize]
      const preds = this.model.predict(inputTensor);
      const arr = await preds.data();
      inputTensor.dispose(); preds.dispose();
      // apply temperature sampling
      const nextIdx = sampleFromDistribution(arr, temperature);
      out += this.tokenizer.indexToChar(nextIdx);
      seq.push(nextIdx);
    }
    return out;
  }
}

// helper sampling
function sampleFromDistribution(probsArray, temperature=1.0) {
  // softmax temperature adjustment
  const p = Float64Array.from(probsArray);
  if (temperature !== 1.0) {
    for (let i=0;i<p.length;i++) p[i] = Math.pow(p[i], 1.0/temperature);
  }
  let sum=0; for (let i=0;i<p.length;i++) sum+=p[i];
  if (sum===0) return 0;
  for (let i=0;i<p.length;i++) p[i]=p[i]/sum;
  const r = Math.random();
  let acc=0;
  for (let i=0;i<p.length;i++){
    acc += p[i];
    if (r <= acc) return i;
  }
  return p.length-1;
}
