class SequenceGenerator {
  generateFibonacciByIndices(startIndex, endIndex) {
    if (startIndex < 0 || endIndex < 0) {
      throw new Error("Indices must be non-negative.");
    }
    if (startIndex > endIndex) {
      throw new Error("Start index cannot be greater than end index.");
    }

    const fibonacciNumbers = [];
    let a = 0,
      b = 1;

    for (let i = 0; i <= endIndex; i++) {
      if (i >= startIndex) {
        fibonacciNumbers.push(a);
      }
      const next = a + b;
      a = b;
      b = next;
    }

    return fibonacciNumbers;
  }

  generate() {
    return [1, 4, 9, 16];
  }
}

let epochs = 0;

const INPUT_SIZE = 11;
const HIDDEN_LAYER = 11;
const CONTEXT_LAYER = 11;
const OUTPUT_SIZE = 1;
const COUNT_OF_PREDICTED_NUMBERS = 3;
const ERROR = 0.0001;
const HORIZONTAL_LAYERS = 2;

const N = 8;
const M = 8;
let width = 0;
let height = 0;

const WIDTH = 256;
const HEIGHT = 256;

class Program {
  constructor() {
    const sequenceGenerator = new SequenceGenerator();

    const sequence = sequenceGenerator.generateFibonacciByIndices(0, 9);

    this.NeuralNetwork = new NeuralNetwork();

    for (let i = 0; i < COUNT_OF_PREDICTED_NUMBERS; i++) {
      const nextNumberEthalon = sequenceGenerator.generateFibonacciByIndices(INPUT_SIZE + i, INPUT_SIZE + i)[0];

      const inputSequence = [...sequence, nextNumberEthalon];

      this.NeuralNetwork.train(inputSequence, nextNumberEthalon);

      const predicted = this.NeuralNetwork.getRestoredValue();

      console.log(predicted);

      sequence.shift();
      sequence.push(predicted);
    }
  }

  start() {}
}

const CMAX = 255;

class NeuralNetwork {
  firstLayerWeights = [];
  secondLayerWeights = [];

  context_layer = [];

  iters = 0;

  weights = [];

  ethalon = null;

  constructor(learningRate = 0.001) {
    this.learningRate = learningRate;

    // this.generateRandomWeights(INPUT_SIZE, OUTPUT_SIZE);

    // this.secondLayerWeights = transposeMattrix(this.firstLayerWeights);

    // this.firstLayerWeights = this.normalize(this.firstLayerWeights)

    // this.secondLayerWeights = this.normalize(this.secondLayerWeights)

    this.weights = this.initWeights();

    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = this.normalize(this.weights[i]);
    }

    // this.firstLayerWeights.map(vector => this.normalizeVector(vector))
    // this.secondLayerWeights.map(vector => this.normalizeVector(vector))
  }

  train(sequence, ethalon) {
    this.ethalon = [ethalon];
    let iters = 0;
    let error = 1;
    let errorsPerIter = 0;
    while (error > ERROR) {
      if (iters !== 0) {
        this.weights[0] = this.context_layer;
      }

      let feedForwardedMattrixes = this.feedForward(sequence);
      error = this.backPropagate(this.ethalon, feedForwardedMattrixes);

      error = errorsPerIter;
      console.log(error);
      errorsPerIter = 0;
      iters++;
    }
    console.log("ITERATIONS: ", iters);
  }

  getRestoredValue(vectorOfColor) {
    return this.feedForward(vectorOfColor).secondLayerMattrix;
  }

  feedForward(vectorOfColor) {
    vectorOfColor = [vectorOfColor];
    // console.log(vectorOfColor, this.firstLayerWeights)
    let firstLayerMattrix = multiplyMatrices(vectorOfColor, this.weights[0]);
    // console.log(firstLayerMattrix)
    let secondLayerMattrix = multiplyMatrices(firstLayerMattrix, this.weights[1]);

    this.context_layer = secondLayerMattrix;

    return {
      firstLayerMattrix,
      secondLayerMattrix,
    };
  }

  backPropagate(vectorOfColor, forwardedMatrices) {
    let { firstLayerMattrix, secondLayerMattrix } = forwardedMatrices;
    // console.log(secondLayerMattrix)

    vectorOfColor = [vectorOfColor];

    let deltaLastLayer = subtractMatrices(secondLayerMattrix, vectorOfColor);

    this.weights[1] = subtractMatrices(
      this.weights[1],
      transposeMattrix(
        multiplyByNumber(multiplyMatrices(transposeMattrix(deltaLastLayer), firstLayerMattrix), this.learningRate)
      )
    );

    let deltaFirstLayer = multiplyMatrices(deltaLastLayer, transposeMattrix(this.weights[1]));

    console.log(this.weights[0], "first layer");
    console.log(
      multiplyByNumber(multiplyMatrices(transposeMattrix(vectorOfColor), deltaFirstLayer), this.learningRate),
      "DSadas"
    );

    this.weights[0] = subtractMatrices(
      this.weights[0],
      multiplyByNumber(multiplyMatrices(transposeMattrix(vectorOfColor), deltaFirstLayer), this.learningRate)
    );

    // this.firstLayerWeights = this.normalize(this.firstLayerWeights)
    // this.secondLayerWeights = this.normalize(this.secondLayerWeights)

    return deltaLastLayer.reduce((acc, row) => acc + row.reduce((sum, val) => sum + val ** 2, 0), 0);
  }

  normalizeVector(vector) {
    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
      let c = vector[i];
      sum += c * c;
    }

    let sqrSum = Math.sqrt(sum);

    if (sqrSum === 0) {
      return vector;
    }

    return vector.map((value) => value / sqrSum);
  }

  generateRandomWeights(inputSize, outputSize) {
    for (let i = 0; i < inputSize; i++) {
      this.firstLayerWeights.push(Array.from({ length: outputSize }, () => Math.random() * 2 - 1));
    }
  }

  normalize(vector) {
    let sum = 0;

    for (let i = 0; i < vector.length; i++) {
      for (let j = 0; j < vector[i].length; j++) {
        let c = vector[i][j];
        sum += c * c;
      }
    }

    let sqrSum = Math.sqrt(sum);

    if (sqrSum === 0) {
      return vector;
    }

    return vector.map((row) => row.map((value) => value / sqrSum));
  }

  initWeights() {
    const weights = [];

    for (let layer = 0; layer < HORIZONTAL_LAYERS; layer++) {
      const weightMatrix = [];

      if (layer === 0) {
        for (let i = 0; i < INPUT_SIZE; i++) {
          const row = [];
          for (let j = 0; j < HIDDEN_LAYER; j++) {
            row.push(Math.random() * 2 - 1);
          }
          weightMatrix.push(row);
        }
      } else {
        for (let i = 0; i < HIDDEN_LAYER; i++) {
          const row = [];
          for (let j = 0; j < OUTPUT_SIZE; j++) {
            row.push(Math.random() * 2 - 1);
          }
          weightMatrix.push(row);
        }
      }

      weights.push(weightMatrix);
    }

    return weights;
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function transposeMattrix(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const transposed = [];

  for (let i = 0; i < cols; i++) {
    transposed[i] = [];
    for (let j = 0; j < rows; j++) {
      transposed[i][j] = matrix[j][i];
    }
  }

  return transposed;
}

function multiplyMatrices(matrixA, matrixB) {
  const rowsA = matrixA.length;
  const colsA = matrixA[0].length;
  const rowsB = matrixB.length;
  const colsB = matrixB[0].length;

  if (colsA !== rowsB) {
    throw new Error("Количество столбцов первой матрицы должно быть равно количеству строк второй матрицы.");
  }

  const result = new Array(rowsA).fill(null).map(() => new Array(colsB).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += matrixA[i][k] * matrixB[k][j];
      }
    }
  }

  return result;
}

function subtractMatrices(matrixA, matrixB) {
  const rowsA = matrixA.length;
  const colsA = matrixA[0].length;
  const rowsB = matrixB.length;
  const colsB = matrixB[0].length;

  if (rowsA !== rowsB || colsA !== colsB) {
    throw new Error("Размеры матриц должны совпадать для выполнения вычитания.");
  }

  const result = new Array(rowsA).fill(null).map(() => new Array(colsA).fill(0));

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsA; j++) {
      result[i][j] = matrixA[i][j] - matrixB[i][j];
    }
  }

  return result;
}

function multiplyByNumber(matrixA, num) {
  return matrixA.map((row) => row.map((el) => el * num));
}

// -------------------------------------------------------------------------------------

class ImageRestorer {
  constructor(bitMaps, blockSize, width, height, canvas) {
    this.bitMaps = bitMaps;
    this.blockSize = blockSize;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = width;
    this.height = height;
  }

  restoreImage() {
    const imageData = this.ctx.createImageData(this.width, this.height);
    const data = imageData.data;

    let blockIndex = 0;
    for (let y = 0; y < this.height; y += this.blockSize) {
      for (let x = 0; x < this.width; x += this.blockSize) {
        const block = this.bitMaps[blockIndex];

        for (let i = 0; i < this.blockSize; i++) {
          for (let j = 0; j < this.blockSize; j++) {
            if (y + i < this.height && x + j < this.width) {
              const pixelIdx = (i * this.blockSize + j) * 3; // Индекс R в block
              const dataIdx = ((y + i) * this.width + (x + j)) * 4; // Индекс R в data

              data[dataIdx] = Math.round(((block[pixelIdx] + 1) * 255) / 2); // R
              data[dataIdx + 1] = Math.round(((block[pixelIdx + 1] + 1) * 255) / 2); // G
              data[dataIdx + 2] = Math.round(((block[pixelIdx + 2] + 1) * 255) / 2); // B
              data[dataIdx + 3] = 255; // Alpha
            }
          }
        }
        blockIndex++;
      }
    }

    // document.getElementById('koeff').append((64 * 1024) / ((64 + 1024) * 64 + 2))
    this.ctx.putImageData(imageData, 0, 0);

    return data;
  }
}

function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

function createChart(ctx, data, title, xLabel, yLabel) {
  // let displayedData = [];
  // console.log(data, 'DATA');
  // const colors = ['red', 'green', 'blue', 'yellow'];
  // for (let i = 0; i < 4; i++) {
  //   displayedData.push({
  //     label: title,
  //     data: data[i],
  //     backgroundColor: colors[i],
  //   });
  // }

  return new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: title,
          data,
          backgroundColor: "blue",
        },
      ],
    },
    options: {
      scales: {
        x: {
          type: "linear",
          position: "bottom",
          title: {
            display: true,
            text: xLabel,
          },
        },
        y: {
          type: "linear",
          position: "left",
          title: {
            display: true,
            text: yLabel,
          },
        },
      },
    },
  });
}

const program = new Program();
