import * as tfvis from '@tensorflow/tfjs-vis';
import { CarsModel } from './models/cars_model';

// 创建我们定义的模型
const model = new CarsModel();

// 输出状态
const statusDiv = document.getElementById('status');
function showStatus(text: string) {
    statusDiv.innerText = text;
}

// 入口
async function run() {
    // 第一步：下载数据、标准化，通过 tfvis 库显示直观的散点图
    showStatus('Downloading and normalizing data...');
    const data = await model.downloadData();
    model.normalizeData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }));
    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'Miles per Gallon',
            height: 300
        }
    );

    // 第二步：创建模型，通过 tfvis 库显示网络层概述
    showStatus('Creating model...');
    model.createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model.model);

    // 第三步：训练模型，通过 tfvis 库生成实时显示 epochs-loss 折线图的回调
    showStatus('Fitting model...');
    const callback = tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
    );
    await model.fitModel(callback);

    // 第四步：测试模型，通过 tfvis 库在同一个图表内用不同颜色显示原始数据和预测数据的散点图以对比
    showStatus('Testing...');
    const {predictedPoints, originalPoints} = model.testModel(data);
    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
        {
            xLabel: 'Horsepower',
            yLabel: 'Miles per Gallon',
            height: 300
        }
    );

    showStatus('Done!');
}

// 网页加载完毕后开始运行
document.addEventListener('DOMContentLoaded', run);
