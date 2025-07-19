const tf = require('@tensorflow/tfjs-node');

// Create a simple neural network using TensorFlow.js
function createModel(inputSize, hiddenSize, outputSize) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: hiddenSize, inputShape: [inputSize], activation: 'relu' }));
  model.add(tf.layers.dense({ units: outputSize, activation: 'sigmoid' }));
  return model;
}

// Train the model using an Adam optimizer and manual gradient updates
async function trainModel(model, xs, ys, epochs, learningRate) {
  const optimizer = tf.train.adam(learningRate);
  for (let i = 0; i < epochs; i++) {
    // Compute gradients and update weights
    optimizer.minimize(() => {
      const preds = model.predict(xs);
      return tf.losses.meanSquaredError(ys, preds);
    });
    if ((i + 1) % 20 === 0) {
      const loss = tf.losses.meanSquaredError(ys, model.predict(xs));
      console.log(`Epoch ${i + 1} loss: ${loss.dataSync()[0].toFixed(4)}`);
    }
  }
}

// Perform Bayesian update given prior and likelihood tensors
function bayesianUpdate(prior, likelihood) {
  const numerator = prior.mul(likelihood);
  const denominator = numerator.add(prior.mul(-1).add(1).mul(likelihood.mul(-1).add(1)));
  return numerator.div(denominator);
}

async function main() {
  // XOR style training data
  const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const ys = tf.tensor2d([[0], [1], [1], [0]]);

  const model = createModel(2, 8, 1);
  await trainModel(model, xs, ys, 200, 0.05);

  const predictions = model.predict(xs);
  predictions.print();

  // Prior probabilities for Bayesian update (assume 0.5)
  const prior = tf.fill([4, 1], 0.5);
  const posterior = bayesianUpdate(prior, predictions);

  console.log('Posterior beliefs after Bayesian update:');
  posterior.print();
}

main();
