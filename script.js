async function getData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter(car => (car.mpg != null && car.horsepower != null));
    return cleaned;
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({inputShape: [1], units: 6, activation: 'sigmoid'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const inputs = data.map(d => d.horsepower);
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max(), inputMin = inputTensor.min();
        const labelMax = labelTensor.max(), labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.adam(0.1),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 50;
    const epochs = 100;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    })

    const predictedPoints = Array.from(xs).map((val, i) => ({
        x: val, y: preds[i]
    }));
    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}

async function run() {
    // Load and plot the original input
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary', model});

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare
    testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);
