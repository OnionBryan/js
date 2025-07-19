import * as tf from '@tensorflow/tfjs-node';

// Generate synthetic data: features are 2D points, labels 0 or 1
const dataSize = 100;
const xsArray = [];
const ysArray = [];
for (let i = 0; i < dataSize; i++) {
  const x1 = Math.random();
  const x2 = Math.random();
  xsArray.push([x1, x2]);
  ysArray.push(x1 > x2 ? 1 : 0); // label depends on comparison
}

// Convert to tensors
const xs = tf.tensor2d(xsArray);
const ys = tf.tensor2d(ysArray, [dataSize, 1]);

// Build a simple model using Adam optimizer
const model = tf.sequential();
model.add(tf.layers.dense({ units: 4, activation: 'relu', inputShape: [2] }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

model.compile({ optimizer: tf.train.adam(0.05), loss: 'binaryCrossentropy' });

// Train using a custom gradient loop for demonstration
async function trainModel() {
  const epochs = 50;
  for (let i = 0; i < epochs; i++) {
    const h = await model.fit(xs, ys, { epochs: 1, shuffle: true });
    console.log(`Epoch ${i + 1}: loss=${h.history.loss[0].toFixed(4)}`);
  }
}

// Simple Bayesian update: prior 0.5 updated by NN prediction as likelihood
function bayesianUpdate(prior, likelihood) {
  const marginal = likelihood * prior + (1 - likelihood) * (1 - prior);
  return marginal > 0 ? (likelihood * prior) / marginal : prior;
}

async function run() {
  await trainModel();
  const testPoint = tf.tensor2d([[0.3, 0.7]]);
  const prediction = model.predict(testPoint);
  const prob = prediction.dataSync()[0];
  console.log('NN prediction probability:', prob.toFixed(4));

  let belief = 0.5; // prior
  belief = bayesianUpdate(belief, prob);
  console.log('Posterior belief:', belief.toFixed(4));
}

run();
