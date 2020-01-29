import * as tf from '@tensorflow/tfjs';
import {BaseCallback, CustomCallbackArgs} from "@tensorflow/tfjs-layers/src/base_callbacks";

const DEFAULT_DATA_URL = 'https://storage.googleapis.com/tfjs-tutorials/carsData.json';

interface Point {
    x: number;
    y: number;
}

interface CarDataItem {
    mpg: number;
    horsepower: number;
}

interface TestResult {
    predictedPoints: Array<Point>;
    originalPoints: Array<Point>;
}

interface NormalizedData<T extends tf.Tensor> {
    inputs: T;
    labels: T;
    inputMax: T;
    inputMin: T;
    labelMax: T;
    labelMin: T;
}

// 模型定义
export class CarsModel {
    // 运行变量
    data: Array<CarDataItem>;
    normalizationData: NormalizedData<tf.Tensor2D>;
    model: tf.Sequential;

    // 部分模型参数
    dataUrl: string;
    optimizer = () => tf.train.adam(0.1);
    loss = tf.losses.meanSquaredError;
    batchSize: number = 50;
    epochs: number = 100;

    constructor(dataUrl?: string) {
        this.dataUrl = dataUrl || DEFAULT_DATA_URL;
    }

    // 从 dataUrl 下载训练模型用的汽车数据，并简化它们
    async downloadData() {
        const carsDataReq = await fetch(this.dataUrl);
        const carsData = await carsDataReq.json();
        this.data = carsData.map((car: any) => ({
            mpg: car["Miles_per_Gallon"],
            horsepower: car["Horsepower"]
        }));
        return this.data;
    }

    // 创建模型、添加神经网络层
    createModel() {
        let model = tf.sequential();
        // Create input layer and hidden layer
        model.add(tf.layers.dense({inputShape: [1], units: 6, activation: 'sigmoid'}));
        // Create output layer
        model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

        this.model = model;
    }

    // 标准化数据
    normalizeData() {
        this.normalizationData = <NormalizedData<tf.Tensor2D>>tf.tidy(() => {
            tf.util.shuffle(this.data);

            const inputs = this.data.map(d => d.horsepower);
            const labels = this.data.map(d => d.mpg);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

            const inputMax = inputTensor.max();
            const inputMin = inputTensor.min();
            const labelMax = labelTensor.max();
            const labelMin = labelTensor.min();

            const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
            const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

            return {
                inputs: normalizedInputs,
                labels: normalizedLabels,
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            };
        });
    }

    // 训练模型
    async fitModel(callbacks?: BaseCallback[]|CustomCallbackArgs|CustomCallbackArgs[]) {
        this.model.compile({
            optimizer: this.optimizer(),
            loss: this.loss,
            metrics: ['mse'],
        });

        const {inputs, labels} = this.normalizationData;

        return await this.model.fit(inputs, labels, {
            batchSize: this.batchSize,
            epochs: this.epochs,
            shuffle: true,
            callbacks
        });
    }

    // 测试模型
    testModel(inputData: Array<CarDataItem>): TestResult {
        const {inputMax, inputMin, labelMax, labelMin} = this.normalizationData;

        const [xs, preds] = tf.tidy(() => {
            const xs = tf.linspace(0, 1, 100);
            const preds = <tf.Tensor>this.model.predict(xs.reshape([100, 1]));

            const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
            const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin);

            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });

        const predictedPoints = Array.from(xs).map((val, i) => ({
            x: val, y: preds[i]
        }));
        const originalPoints = inputData.map(d => ({
            x: d.horsepower, y: d.mpg,
        }));

        return {predictedPoints, originalPoints};
    }

}
